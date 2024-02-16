import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from utils.classifier import ClassifierLightning
from utils.data import MILDataset, get_multi_cohort_df
from utils.options import Options
from utils.utils import save_results

"""
evaluate slide-level classification task for
 - given feature_dir
 - given dataset
train, validate, and test a model with nested k-fold cross validation with in-domain test set and external test set.

k-fold cross validation (k=5)
[--|--|--|**|##]
[--|--|**|##|--]
[--|**|##|--|--]
[**|##|--|--|--]
[##|--|--|--|**]
where 
-- train
** val
## test
"""


# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)


def eval_slide_level(cfg):
    cfg.seed = torch.randint(0, 1000, (1,)).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    # --------------------------------------------------------
    # set up paths
    # --------------------------------------------------------

    # saving locations
    base_path = Path(cfg.save_dir)  # adapt to own target path
    cfg.logging_name = (
        f'{cfg.name}_{cfg.model}_{"-".join(cfg.cohorts)}_{cfg.norm}_{cfg.target}' if cfg.name != "debug" else "debug"
    )
    base_path = base_path / cfg.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / "models"
    result_path = base_path / "results"
    result_path.mkdir(parents=True, exist_ok=True)

    # check if results csv exists already
    if Path(base_path / f"results_test_{cfg.logging_name}.csv").exists() and cfg.name != "debug":
        print(f"results for {cfg.logging_name} already exist")
        return

    norm_val = "raw" if cfg.norm in ["histaugan", "efficient_histaugan"] else cfg.norm
    norm_test = "raw" if cfg.norm in ["histaugan", "efficient_histaugan"] else cfg.norm

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------

    print("\n--- load dataset ---")
    data, clini_info = get_multi_cohort_df(
        cfg.data_config,
        cfg.cohorts,
        [cfg.target],
        cfg.label_dict,
        norm=cfg.norm,
        feats=cfg.feats,
        clini_info=cfg.clini_info,
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    for cohort in cfg.cohorts:
        if cohort in cfg.ext_cohorts:
            cfg.ext_cohorts.pop(cfg.ext_cohorts.index(cohort))

    train_cohorts = f'{", ".join(cfg.cohorts)}'
    test_cohorts = [train_cohorts, *cfg.ext_cohorts]
    results = {t: [] for t in test_cohorts}

    # external test set
    test_ext_dataloader = []
    for ext in cfg.ext_cohorts:
        test_data, clini_info = get_multi_cohort_df(
            cfg.data_config,
            [ext],
            [cfg.target],
            cfg.label_dict,
            norm=norm_test,
            feats=cfg.feats,
            clini_info=cfg.clini_info,
        )
        dataset_ext = MILDataset(
            test_data,
            test_data["PATIENT"],
            [cfg.target],
            clini_info=clini_info,
            norm=norm_test,
        )
        test_ext_dataloader.append(
            DataLoader(
                dataset=dataset_ext,
                batch_size=1,
                shuffle=False,
                num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                pin_memory=True,
            )
        )

    print(f"training cohorts: {train_cohorts}")
    print(f"testing cohorts:  {cfg.ext_cohorts}")

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------

    # load fold directory from data_config
    fold_path = Path(args.data_config[train_cohorts]["folds"]) / f"{cfg.target}_{cfg.folds}folds"
    fold_path.mkdir(parents=True, exist_ok=True)

    # split data stratified by the labels
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    patient_df = data.groupby("PATIENT").first().reset_index()
    target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    splits = skf.split(patient_df, patient_df[target_stratisfy])
    splits = list(splits)

    for k in range(cfg.folds):
        # read split from csv-file if exists already else save split to csv
        if Path(fold_path / f"fold{k}_train.csv").exists():
            train_idxs = pd.read_csv(fold_path / f"fold{k}_train.csv", index_col="Unnamed: 0")
            val_idxs = pd.read_csv(fold_path / f"fold{k}_val.csv", index_col="Unnamed: 0")
            test_idxs = pd.read_csv(fold_path / f"fold{k}_test.csv", index_col="Unnamed: 0")
            # train_idxs = np.loadtxt(fold_path / f'fold{k}_train.csv', dtype=str).tolist()
            # val_idxs = np.loadtxt(fold_path / f'fold{k}_val.csv', dtype=str).tolist()
            # test_idxs = np.loadtxt(fold_path / f'fold{k}_test.csv', dtype=str).tolist()
        else:
            train_idxs, val_idxs = train_test_split(
                splits[k][0], stratify=patient_df.iloc[splits[k][0]][target_stratisfy], random_state=cfg.seed
            )
            train_idxs = patient_df["PATIENT"].iloc[train_idxs]
            val_idxs = patient_df["PATIENT"].iloc[val_idxs]
            test_idxs = patient_df["PATIENT"].iloc[splits[k][1]]
            train_idxs.to_csv(fold_path / f"folds{k}_train.csv")
            val_idxs.to_csv(fold_path / f"folds{k}_val.csv")
            test_idxs.to_csv(fold_path / f"folds{k}_test.csv")

        # training dataset
        train_dataset = MILDataset(
            data,
            train_idxs,
            [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm,
            clini_info=cfg.clini_info,
        )
        print(f"num training samples in fold {k}: {len(train_dataset)}")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.bs,
            shuffle=True,
            num_workers=0, # int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            pin_memory=True,
        )
        if len(train_dataloader) < cfg.val_check_interval:
            cfg.val_check_interval = len(train_dataloader)
        if cfg.lr_scheduler == "OneCycleLR":
            cfg.lr_scheduler_config["total_steps"] = cfg.num_epochs * len(train_dataloader)

        # validation dataset
        val_dataset = MILDataset(data, val_idxs, [cfg.target], norm=norm_val, clini_info=cfg.clini_info)
        print(f"num validation samples in fold {k}: {len(val_dataset)}")
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0, # int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            pin_memory=True,
        )

        # test dataset (in-domain)
        test_dataset = MILDataset(data, test_idxs, [cfg.target], norm=norm_test, clini_info=cfg.clini_info)
        print(f"num test samples in fold {k}: {len(test_dataset)}")
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            pin_memory=True,
        )

        # set class weighting for binary classification
        if cfg.task == "binary":
            num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])
            cfg.pos_weight = torch.Tensor((len(train_dataset) - num_pos) / num_pos)

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        model = ClassifierLightning(cfg)

        # --------------------------------------------------------
        # logging
        # --------------------------------------------------------
        logger = WandbLogger(
            project=cfg.project,
            entity="histo-collab",
            name=f"{cfg.logging_name}_fold{k}",
            group=f"{cfg.logging_name}",
            tags=[f"{cfg.cohorts}"],
            save_dir=cfg.save_dir,
            reinit=True,
            settings=wandb.Settings(start_method="fork"),
            mode="online",
        )

        csv_logger = CSVLogger(
            save_dir=result_path,
            name=f"fold{k}",
        )

        # --------------------------------------------------------
        # callbacks
        # --------------------------------------------------------
        checkpoint_callback = ModelCheckpoint(
            monitor="auroc/val" if cfg.stop_criterion == "auroc" else "loss/val",
            dirpath=model_path,
            filename=f"best_model_{cfg.logging_name}_fold{k}",
            save_top_k=1,
            mode="max" if cfg.stop_criterion == "auroc" else "min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        # --------------------------------------------------------
        # set up trainer
        # --------------------------------------------------------

        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            accelerator="auto",
            devices=-1,
            callbacks=[checkpoint_callback, lr_monitor],
            max_epochs=cfg.num_epochs,
            val_check_interval=cfg.val_check_interval,
            check_val_every_n_epoch=None,
            enable_model_summary=False,
        )

        # --------------------------------------------------------
        # training
        # --------------------------------------------------------

        if Path(model_path / f"best_model_{cfg.logging_name}_fold{k}.pth").exists():
            pass
        else:
            results_val = trainer.fit(
                model,
                train_dataloader,
                val_dataloader,
            )
            logger.log_table("results/val", results_val)

        # --------------------------------------------------------
        # testing
        # --------------------------------------------------------

        test_cohorts_dataloader = [test_dataloader, *test_ext_dataloader]
        for idx in range(len(test_cohorts)):
            print("Testing: ", test_cohorts[idx])
            results_test = trainer.test(
                model,
                test_cohorts_dataloader[idx],
                ckpt_path="best",
            )
            results[test_cohorts[idx]].append(results_test[0])
            # save patient predictions to outputs csv file
            model.outputs.to_csv(result_path / f"fold{k}" / f"outputs_{test_cohorts[idx]}.csv")

        wandb.finish()  # required for new wandb run in next fold
        torch.cuda.empty_cache()

    # save results to csv file
    save_results(cfg, results, base_path, train_cohorts, test_cohorts)


if __name__ == "__main__":
    parser = Options()
    args = parser.parse()

    # uses all subdirectories of the given feature directory if it is named "features"
    if args.feature_dir is None:
        args.feature_dir = Path(args.base_dir) / "features"

    if Path(args.feature_dir).name == "features":
        feature_dirs = [f.path for f in os.scandir(args.feature_dir) if f.is_dir()]
    else:
        feature_dirs = [args.feature_dir]
    
    # set input dim depending on model
    input_dims = {
        "ctranspath": 768,
        "owkin": 768,
        "vim_finetuned": 192,
        "dinov2_finetuned": 384,
        "dinov2_vits14_downloaded": 384,
        "resnet50": 1024,
        "resnet50full": 2048,
    }

    # retrieve task/dataset from feature directory
    configs = [Path(f).name.split("_")[0] for f in feature_dirs]
    if "CPTAC-CRC" in configs:
        configs.pop(configs.index("CPTAC-CRC"))
        feature_dirs.pop(feature_dirs.index([f for f in feature_dirs if "CPTAC-CRC" in f][0]))
    
    for fd, c in zip(feature_dirs, configs):

        with open(args.data_config, "r") as f:
            args.data_config = yaml.safe_load(f)

        args.save_dir = Path(args.base_dir) / 'results'
        args.name = Path(fd).name if args.name is None else args.name
        
        args.feats = "custom"
        args.data_config[c]["feature_dir"]["raw"][args.feats] = fd
        if c == "TCGA-CRC":
            args.data_config["CPTAC-CRC"]["feature_dir"]["raw"][args.feats] = fd.replace("TCGA-CRC", "CPTAC-CRC")

        # set input dim depending on model
        args.input_dim = input_dims[args.feature_extractor]

        # Load the configuration from the YAML file
        args.config_file = f"dinov2/eval/slide_level/configs/{c}.yaml"
        with open(args.config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Update the configuration with the values from the argument parser
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None and arg_name != "config_file":
                config[arg_name] = getattr(args, arg_name)

        print("\n--- load options ---")
        for name, value in sorted(config.items()):
            print(f"{name}: {str(value)}")

        config = argparse.Namespace(**config)
        eval_slide_level(config)
