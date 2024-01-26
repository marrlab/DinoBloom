import torch
import torch.nn as nn
from torchvision import transforms

from torchvision.models import resnet
from models.ctran import ctranspath
from models.resnet_retccl import resnet50 as retccl_res50
from models.sam import build_sam_vit_h,build_sam_vit_b,build_sam_vit_l
from models.imagebind import imagebind_huge
from transformers import  Data2VecVisionModel, BeitFeatureExtractor
from models.dinov2 import vit_small, vit_base, vit_large, vit_giant2
from models.vim import get_vision_mamba_model
import math
import torch.nn.functional as nnf
#import tensorflow_hub as hub
#import tensorflow as tf

RETCCL_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/retccl.pth'
CTRANSPATH_PATH = '/lustre/groups/shared/users/peng_marr/pretrained_models/ctranspath.pth'
SAM_VIT_H_PATH='/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_h.pth'
SAM_VIT_L_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_l.pth"
SAM_VIT_B_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/sam_vit_b_01ec64.pth"
DINO_VIT_S_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vits14_pretrain.pth"
DINO_VIT_B_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitb14_pretrain.pth"
DINO_VIT_L_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitl14_pretrain.pth"
DINO_VIT_G_PATH="/lustre/groups/shared/users/peng_marr/pretrained_models/dinov2_vitg14_pretrain.pth"
DINO_VIT_S_PATH_FINETUNED="/home/icb/valentin.koch/dinov2/debug/eval/training_12499/teacher_checkpoint_mlp.pth"
DINO_VIT_S_PATH_FINETUNED_DOWNLOADED="/lustre/scratch/users/benedikt.roth/dinov2_vits_interpolated_224_NCT-CRC_downloaded_model_finetuned_10000k_iterations/eval/training_2699/teacher_checkpoint.pth"
#DINO_VIT_S_PATH_FINETUNED_DOWNLOADED="/lustre/scratch/users/benedikt.roth/dinov2_vitg_interpolated_224_NCT-CRC_downloaded_model_finetuned/eval/training_119999/teacher_checkpoint.pth"

def get_models(modelname, checkpoint=None):
    models = []
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    if modelname.lower() == 'ctranspath':
        model = get_ctranspath()
    #elif modelname.lower() == 'remedis':
    #    model = hub.load('cxr-52x2-remedis-m')
    elif modelname.lower() == 'resnet50':
        model = get_res50()
    elif modelname.lower() == 'retccl':
        model = get_retCCL()
    elif modelname.lower() == 'resnet50_full':
        model = get_full_res50()
    elif modelname.lower()=="sam_vit_h":
        model=get_sam_vit_h()
    elif modelname.lower()=="sam_vit_b":
        model=get_sam_vit_b()
    elif modelname.lower()=="sam_vit_l":
        model=get_sam_vit_l()
    elif modelname.lower() == 'dinov2_vits14':
        model=get_dino_vit_s()
    elif modelname.lower() == 'dinov2_vits14_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif modelname.lower() == 'dinov2_vits14_reg_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    elif modelname.lower() == 'dinov2_vitb14':
        model=get_dino_vit_b()
    elif modelname.lower() == 'dinov2_vitb14_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif modelname.lower() == 'dinov2_vitl14':
        model=get_dino_vit_l()
    elif modelname.lower() == 'dinov2_vitl14_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif modelname.lower() == 'dinov2_vitg14':
        model=get_dino_vit_g()
    elif modelname.lower() == 'dinov2_vitg14_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    elif modelname.lower() == 'dinov2_vitg14_reg_downloaded':
        model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    elif modelname.lower() == 'dinov2_finetuned':
        model=get_dino_finetuned()
    elif modelname.lower() == 'vim_finetuned':
        model=get_vim_finetuned(checkpoint=checkpoint)
    elif modelname.lower() == 'dinov2_finetuned_downloaded':
        model=get_dino_finetuned_downloaded()
    elif modelname.lower() == 'dinov2_vits14_interpolated':
            model = get_dino_vit_s_interpolated()
    elif modelname.lower()=="imagebind":
        model=get_imagebind()
    elif modelname.lower()=='beit_fb':
        model = BeitModel(device)
    model = model.to(device)
    model.eval()
    return model

def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    patch_pos_embed = nnf.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)

'''
# for 518
def get_dino_finetuned():
    vit_kwargs = dict(
        img_size=518,
        patch_size=14,
        #init_values=1.0,
        #ffn_layer="mlp",
        #block_chunks=0,
    )
    model = vit_small(**vit_kwargs)
    pretrained = torch.load(DINO_VIT_S_PATH_FINETUNED)
    new_state_dict = {}

    for key, value in pretrained['teacher'].items():
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    return model
'''

def get_retCCL():
    model = retccl_res50(num_classes=128, mlp=False,
                         two_branch=False, normlinear=True)
    pretext_model = torch.load(RETCCL_PATH, map_location=torch.device('cpu'))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model

#for 224
def get_dino_finetuned():
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        #init_values=1.0,
        #ffn_layer="mlp",
        #block_chunks=0,
    )
    model = vit_small(**vit_kwargs)
    pretrained = torch.load(DINO_VIT_S_PATH_FINETUNED)
    new_state_dict = {}

    for key, value in pretrained['teacher'].items():
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

def get_vim_finetuned(checkpoint=None):
    model = get_vision_mamba_model(checkpoint=checkpoint)
    return model

#for 224
def get_dino_finetuned_downloaded():
    # pos_embed has wrong shape
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    # load finetuned weights
    pretrained = torch.load(DINO_VIT_S_PATH_FINETUNED_DOWNLOADED, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key:
            print('not used')
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value
    #change shape of pos_embed
    pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
    #pos_embed = nn.Parameter(torch.zeros(1, 257, 1536))
    model.pos_embed = pos_embed
    # load state dict
    model.load_state_dict(new_state_dict, strict=True)
    return model


def get_sam_vit_h():
    return build_sam_vit_h(SAM_VIT_H_PATH)

def get_sam_vit_l():
    return build_sam_vit_l(SAM_VIT_L_PATH)

def get_sam_vit_b():
    return build_sam_vit_b(SAM_VIT_B_PATH)

#def get_dino_vit_s():
#    vit_kwargs = dict(
#        img_size=518,
#        patch_size=14,
#        #init_values=1.0,
#        #ffn_layer="mlp",
#        #block_chunks=0,
#    )
#    model = vit_small(**vit_kwargs)
#    chkpt = torch.load(DINO_VIT_S_PATH)
#    chkpt_2 = interpolate_pos_encoding(chkpt['pos_embed'], 16, 16)
#    chkpt['pos_embed'] = chkpt_2
#    model.load_state_dict(chkpt, strict=False)
#    return model

def get_dino_vit_s_interpolated():
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    input_tensor = model.pos_embed
    tensor_corr_shape = interpolate_pos_encoding(input_tensor, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed

    #test old interpolation
    
    #vit_kwargs = dict(
    #    img_size=224,
    #    patch_size=14,
    #    #init_values=1.0,
    #    #ffn_layer="mlp",
    #    #block_chunks=0,
    #)
    #model_test = vit_small(**vit_kwargs)
    #chkpt = torch.load(DINO_VIT_S_PATH)
    #chkpt_2 = interpolate_pos_encoding(chkpt['pos_embed'], 16, 16)
    #chkpt['pos_embed'] = chkpt_2
    #model_test.load_state_dict(chkpt, strict=False)
    
    return model

def get_dino_vit_s():
    #model_test = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #chkpt = torch.load(DINO_VIT_S_PATH)
    #model_test.load_state_dict(chkpt, strict=False)
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    input_tensor = model.pos_embed
    class_token = input_tensor[:, 0:1, :]
    rest = input_tensor[:, 1:, :]

    reshaped_tensor = rest.view(1, 37, 37, 384)

    middle = 18
    middle_start = middle - 8
    middle_end = middle + 8
    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
    flattened_tensor = middle_part.reshape(1, 256, 384)

    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)

    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape

    model.pos_embed = pos_embed
    
    return model

#def get_dino_vit_b():
#    vit_kwargs = dict(
#        img_size=224,
#        patch_size=14,
#        #init_values=1.0,
#        #ffn_layer="mlp",
#        #block_chunks=0,
#    )
#    model = vit_base(**vit_kwargs)
#    chkpt = torch.load(DINO_VIT_B_PATH)
#    #chkpt_2 = interpolate_pos_encoding(chkpt['pos_embed'], 16, 16)
#    #chkpt['pos_embed'] = chkpt_2
#
#    input_tensor = chkpt['pos_embed']
#    class_token = input_tensor[:, 0:1, :]
#    rest = input_tensor[:, 1:, :]
#
#    reshaped_tensor = rest.view(1, 37, 37, 768)
#
#    middle = 19
#    middle_start = middle - 8
#    middle_end = middle + 8
#    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
#
#    flattened_tensor = middle_part.reshape(1, 256, 768)
#
#    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)
#    chkpt['pos_embed'] = tensor_corr_shape
#
#    model.load_state_dict(chkpt, strict=False)
#    return model

def get_dino_vit_b():
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    input_tensor = model.pos_embed
    class_token = input_tensor[:, 0:1, :]
    rest = input_tensor[:, 1:, :]

    reshaped_tensor = rest.view(1, 37, 37, 768)

    middle = 18
    middle_start = middle - 8
    middle_end = middle + 8
    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
    flattened_tensor = middle_part.reshape(1, 256, 768)

    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)

    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape

    model.pos_embed = pos_embed
    
    return model

#def get_dino_vit_l():
#    vit_kwargs = dict(
#        img_size=224,
#        patch_size=14,
#        #init_values=1.0,
#        #ffn_layer="mlp",
#        #block_chunks=0,
#    )
#    model = vit_large(**vit_kwargs)
#    chkpt = torch.load(DINO_VIT_L_PATH)
#    #chkpt_2 = interpolate_pos_encoding(chkpt['pos_embed'], 16, 16)
#    #chkpt['pos_embed'] = chkpt_2
#
#    input_tensor = chkpt['pos_embed']
#    class_token = input_tensor[:, 0:1, :]
#    rest = input_tensor[:, 1:, :]
#
#    reshaped_tensor = rest.view(1, 37, 37, 1024)
#
#    middle = 19
#    middle_start = middle - 8
#    middle_end = middle + 8
#    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
#
#    flattened_tensor = middle_part.reshape(1, 256, 1024)
#
#    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)
#    chkpt['pos_embed'] = tensor_corr_shape
#
#    model.load_state_dict(chkpt, strict=False)
#    return model


def get_dino_vit_l():
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    input_tensor = model.pos_embed
    class_token = input_tensor[:, 0:1, :]
    rest = input_tensor[:, 1:, :]

    reshaped_tensor = rest.view(1, 37, 37, 1024)

    middle = 18
    middle_start = middle - 8
    middle_end = middle + 8
    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
    flattened_tensor = middle_part.reshape(1, 256, 1024)

    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)

    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape

    model.pos_embed = pos_embed
    
    return model

def get_dino_vit_g():
    model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    input_tensor = model.pos_embed
    class_token = input_tensor[:, 0:1, :]
    rest = input_tensor[:, 1:, :]

    reshaped_tensor = rest.view(1, 37, 37, 1536)

    middle = 18
    middle_start = middle - 8
    middle_end = middle + 8
    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
    flattened_tensor = middle_part.reshape(1, 256, 1536)

    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)

    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape

    model.pos_embed = pos_embed
    
    return model

#def get_dino_vit_g():
#    vit_kwargs = dict(
#        img_size=518,
#        patch_size=14,
#        init_values=1.0,
#        ffn_layer="mlp",
#        block_chunks=0,
#    )
#    model = vit_giant2(**vit_kwargs)
#    pretrained = torch.load(DINO_VIT_G_PATH)
#    '''
#    input_tensor = pretrained['pos_embed']
#    class_token = input_tensor[:, 0:1, :]
#    rest = input_tensor[:, 1:, :]
#
#    reshaped_tensor = rest.view(1, 37, 37, 1536)
#
#    middle = 18
#    middle_start = middle - 8
#    middle_end = middle + 8
#    middle_part = reshaped_tensor[:, middle_start:middle_end, middle_start:middle_end, :]
#    flattened_tensor = middle_part.reshape(1, 256, 1536)
#
#    tensor_corr_shape = torch.cat((class_token, flattened_tensor), dim=1)
#    pretrained['pos_embed'] = tensor_corr_shape
#    '''
#    model.load_state_dict(pretrained, strict=False)
#    return model


def get_ctranspath():
    model = ctranspath()
    model.head = nn.Identity()
    pretrained = torch.load(CTRANSPATH_PATH)
    model.load_state_dict(pretrained['model'], strict=True)
    return model

def get_res50():

    model = resnet.resnet50(weights='ResNet50_Weights.DEFAULT')

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)
        
    #delete last res block as this has been shown to work better
    model = nn.Sequential(*list(model.children())
                          [:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

    return model

def get_full_res50():
    model = resnet.resnet50(weights='ResNet50_Weights.DEFAULT')
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

    if model_name.lower() in ['ctranspath', 'resnet50',"simclr_lung", 'beit_fb', 'resnet50_full']:
        resolution = 224
    elif model_name.lower() in ['dinov2_vitg14_downloaded','dinov2_vits14_downloaded','dinov2_vitb14_downloaded','dinov2_vitl14_downloaded','dinov2_vits14_reg_downloaded','dinov2_vitg14_reg_downloaded']:
        resolution = 518
    elif model_name.lower() == 'retccl':
        resolution = 256
    elif model_name.lower() == 'kimianet':
        resolution = 1000
    elif model_name.lower() == 'imagebind':
        resolution = 224
        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)
    # change later to correct value
    elif model_name.lower() in ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14','dinov2_finetuned','dinov2_vits14_interpolated','dinov2_finetuned_downloaded','remedis', 'vim_finetuned']:
        resolution = 224
    elif "sam" in model_name.lower():
        resolution = 1024
        mean=(123.675, 116.28, 103.53)
        std=(58.395, 57.12, 57.375)
    else:
        raise ValueError('Model name not found')

    transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    if 'beit_fb' in model_name.lower():
        transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
    ]
    
    elif "sam" in model_name.lower():
        # multiply image by 255 for "sam" model
        transforms_list = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Lambda(multiply_by_255),
        transforms.Normalize(mean=mean, std=std),
    ]

    preprocess_transforms = transforms.Compose(transforms_list)
    return preprocess_transforms


class BeitModel(torch.nn.Module):
    def __init__(self, device, pretrained_model='facebook/data2vec-vision-base', image_size=224, patch_size=16):
        super(BeitModel, self).__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(pretrained_model)
        self.image_size = image_size
        self.patch_size = patch_size
        self.model =  Data2VecVisionModel.from_pretrained(pretrained_model)
        self.device=device
        self.avg_pooling=nn.AdaptiveAvgPool1d((1))

    def forward(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        inputs=inputs['pixel_values'].to(self.device)
        outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = outputs.hidden_states

        # The provided code was taking only the 13th layer, I kept that behaviour
        features = encoder_hidden_states[12][:,1:,:].permute(0,2,1)
        features=self.avg_pooling(features)

        return features.squeeze(dim=2)
    

if __name__ == '__main__':
    get_models(['resnet50'])
