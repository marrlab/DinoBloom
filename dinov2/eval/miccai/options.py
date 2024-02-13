import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Feature extraction")

        # model
        self.parser.add_argument(
            "--model", help="select model ctranspath, retccl, all", nargs="+", default="dinov2_finetuned", type=str
        )
        self.parser.add_argument("--run", help="path to folder of run", type=str)
        self.parser.add_argument(
            "--data_config",
            help="path to config file for data paths",
            default="dinov2/eval/slide_level/configs/data_config.yaml",
            type=str,
        )
        self.parser.add_argument("--dataset", help="name of dataset", default="TCGA-CRC", type=str)
        # size and resolution
        self.parser.add_argument("--patch_size", help="Patch size for saving", default=512, type=int)
        self.parser.add_argument(
            "--resolution_in_mpp",
            help="resolution in mpp, usually 10x= 1mpp, 20x=0.5mpp, 40x=0.25, ",
            default=1,
            type=float,
        )
        self.parser.add_argument(
            "--downscaling_factor",
            help="only used if >0, overrides manual resolution. needed if resolution not given",
            default=0,
            type=float,
        )

        # tissue segmentation
        self.parser.add_argument(
            "--white_thresh",
            help="if all RGB pixel values are larger than this value, the pixel is considered as white/background",
            default=[175, 190, 178],
            nargs="+",
            type=int,
        )
        self.parser.add_argument(
            "--black_thresh",
            help="if all RGB pixel values are smaller or equal than this value, the pixel is considered as black/background",
            default=0,
            type=str,
        )
        self.parser.add_argument(
            "--invalid_ratio_thresh", help="maximum acceptable amount of background", default=0.3, type=float
        )
        self.parser.add_argument(
            "--edge_threshold",
            help="canny edge detection threshold. if smaller than this value, patch gets discarded",
            default=1,
            type=int,
        )
        self.parser.add_argument(
            "--calc_thresh",
            help="darker colours than this are considered calc",
            default=[40, 40, 40],
            nargs="+",
            type=int,
        )

        # other options
        self.parser.add_argument(
            "--file_extension", help="file extension the slides are saved under, e.g. tiff", default=".svs", type=str
        )
        self.parser.add_argument(
            "--scene_list", help="list of scene(s) to be extracted", nargs="+", default=[0], type=int
        )
        self.parser.add_argument("--save_tile_preview", help="set True if you want nice pictures", action="store_true")
        self.parser.add_argument(
            "--save_patch_images", help="True if each patch should be saved as an image", action="store_true"
        )
        self.parser.add_argument("--preview_size", help="size of tile_preview", default=4096, type=int)
        self.parser.add_argument("--batch_size", default=256, type=int)
        self.parser.add_argument(
            "--exctraction_list",
            help="if only a subset of the slides should be extracted save their names in a csv",
            default=None,
            type=str,
        )  #
        self.parser.add_argument(
            "--save_qupath_annotation", help="set True if you want nice qupath annotations", default=False, type=bool
        )
        self.parser.add_argument(
            "--split",
            help="(k,n): split slides into n distinct chunks and process number k. (0,1) for all slides at once. E.g. one urn with (0,2) and one with (1,2) to split data.",
            default=[0, 1],
            nargs="+",
            type=int,
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        return self.opt
