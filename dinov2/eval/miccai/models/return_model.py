import torch
import torch.nn as nn
from models.ctran import ctranspath
from models.imagebind import imagebind_huge
from models.resnet_retccl import resnet50 as retccl_res50
from models.sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

from torchvision import transforms
from torchvision.models import resnet
from transformers import BeitFeatureExtractor, Data2VecVisionModel, ViTModel, AutoImageProcessor

# RETCCL_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/retccl.pth'
# CTRANSPATH_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/ctranspath.pth'
# SAM_VIT_H_PATH='/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_h.pth'
# SAM_VIT_L_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_l.pth"
# SAM_VIT_B_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_b_01ec64.pth"
# DINO_VIT_S_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vits14_pretrain.pth"
# DINO_VIT_B_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitb14_pretrain.pth"
# DINO_VIT_L_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitl14_pretrain.pth"
# DINO_VIT_G_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitg14_pretrain.pth"
# DINO_VIT_S_PATH_FINETUNED="/home/icb/valentin.koch/dinov2/debug/eval/training_12499/teacher_checkpoint_mlp.pth"
# DINO_VIT_S_PATH_FINETUNED_DOWNLOADED="/lustre/scratch/users/benedikt.roth/dinov2_vits_interpolated_224_NCT-CRC_downloaded_model_finetuned_10000k_iterations/eval/training_2699/teacher_checkpoint.pth"
# DINO_VIT_S_PATH_FINETUNED_DOWNLOADED="/lustre/scratch/users/benedikt.roth/dinov2_vitg_interpolated_224_NCT-CRC_downloaded_model_finetuned/eval/training_119999/teacher_checkpoint.pth"


def get_models(modelname, saved_model_path=None):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- histology-pretrained models
    if modelname.lower() == "ctranspath":
        model = get_ctranspath(saved_model_path)
    elif modelname.lower() == 'remedis':
        model = hub.load('cxr-52x2-remedis-m')
    elif modelname.lower() == "retccl":
        model = get_retCCL(saved_model_path)
    elif modelname.lower() == "owkin":
        model = Phikon()

    # --- vision foundation models 
    elif modelname.lower() == "resnet50":
        model = get_res50()
    elif modelname.lower() == "resnet50_full":
        model = get_full_res50()

    elif modelname.lower() == "imagebind":
        model = get_imagebind(saved_model_path)
    elif modelname.lower() == "beit_fb":
        model = BeitModel(device)    

    # --- our finetuned models
    elif modelname.lower() in ["dinov2_vits14","dinov2_vitb14","dinov2_vitl14","dinov2_vitg14"]:
        model = get_dino_finetuned_downloaded(saved_model_path, modelname)

    elif modelname.lower() == "vim_finetuned":
        model = get_vim_finetuned(saved_model_path)
        
    else: 
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model


def get_retCCL(model_path):
    model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model


def get_vim_finetuned(checkpoint=None):
    from models.vim import get_vision_mamba_model

    model = get_vision_mamba_model(checkpoint=checkpoint)
    return model

# for 224
def get_dino_finetuned_downloaded(model_path, modelname):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    # load finetuned weights

    # pos_embed has wrong shape
    if model_path is not None:
        pretrained = torch.load(model_path, map_location=torch.device("cpu"))
        # make correct state dict for loading
        new_state_dict = {}
        for key, value in pretrained["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace("backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        pos_embed = nn.Parameter(torch.zeros(1, 257, input_dims[modelname]))
        model.pos_embed = pos_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)
    return model


def get_sam_vit_h(model_path):
    return build_sam_vit_h(model_path)


def get_sam_vit_l(model_path):
    return build_sam_vit_l(model_path)


def get_sam_vit_b(model_path):
    return build_sam_vit_b(model_path)



def get_ctranspath(model_path):
    model = ctranspath()
    model.head = nn.Identity()
    pretrained = torch.load(model_path)
    model.load_state_dict(pretrained["model"], strict=True)
    return model


def get_res50():

    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    # delete last res block as this has been shown to work better
    model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

    return model


def get_full_res50():
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")
    return model


def get_imagebind(pretrained=True):
    model = imagebind_huge(pretrained=pretrained)
    return model


def multiply_by_255(img):
    return img * 255


def get_transforms(model_name):
    # from imagenet, leave as is
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ["ctranspath", "resnet50", "simclr_lung", "beit_fb", "resnet50_full"]:
        size = 224
    elif model_name.lower() == "owkin":
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        mean, std = image_processor.image_mean, image_processor.image_std
        size = image_processor.size['height']
    # elif model_name.lower() in [
    #     "dinov2_vitg14_downloaded",
    #     "dinov2_vits14_downloaded",
    #     "dinov2_vitb14_downloaded",
    #     "dinov2_vitl14_downloaded",
    #     "dinov2_vits14_reg_downloaded",
    #     "dinov2_vitg14_reg_downloaded",
    # ]:
    #     size = 518
    elif model_name.lower() == "retccl":
        size = 256
    elif model_name.lower() == "kimianet":
        size = 1000
    elif model_name.lower() == "imagebind":
        size = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    # change later to correct value
    elif model_name.lower() in [
        "dinov2_vits14",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_finetuned",
        "dinov2_vits14_interpolated",
        "dinov2_finetuned_downloaded",
        "remedis",
        "vim_finetuned",
    ]:
        size = 224
    elif "sam" in model_name.lower():
        size = 1024
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)
    else:
        raise ValueError("Model name not found")
    
    size=(size,size)
    
    transforms_list = [transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

    if "beit_fb" in model_name.lower():
        transforms_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]

    elif "sam" in model_name.lower():
        # multiply image by 255 for "sam" model
        transforms_list = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(multiply_by_255),
            transforms.Normalize(mean=mean, std=std),
        ]

    preprocess_transforms = transforms.Compose(transforms_list)
    return preprocess_transforms


class BeitModel(torch.nn.Module):
    def __init__(self, device, pretrained_model="facebook/data2vec-vision-base", image_size=224, patch_size=16):
        super(BeitModel, self).__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(pretrained_model)
        self.image_size = image_size
        self.patch_size = patch_size
        self.model = Data2VecVisionModel.from_pretrained(pretrained_model)
        self.device = device
        self.avg_pooling = nn.AdaptiveAvgPool1d((1))

    def forward(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs = inputs["pixel_values"].to(self.device)
        outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = outputs.hidden_states

        # The provided code was taking only the 13th layer, I kept that behaviour
        features = encoder_hidden_states[12][:, 1:, :].permute(0, 2, 1)
        features = self.avg_pooling(features)

        return features.squeeze(dim=2)


class Phikon(nn.Module):
    def __init__(self):
        super(Phikon, self).__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]
        return features