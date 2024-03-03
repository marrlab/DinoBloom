import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import torch.nn as nn
import os
import glob
from pathlib import Path



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

patch_h = 16 #14*14
patch_w = 16 # 14*14 14*16=224
feat_dim = 384 # vits14

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

model_path = "/mnt/ceph_vol/dinov2/models/vits_9999.pth"
modelname = "dinov2_vits14"

model = get_dino_finetuned_downloaded(model_path, modelname)

paths=[]

dir_path = "/mnt/ceph_vol/dinov2/dinov2/data/blood_sampels/vis"
files = Path(dir_path).glob("*.png")
print(files)
paths = [os.path.join(dir_path, file) for file in files]
number_of_images=len(paths)
# Modify the list comprehension to move each image tensor to CUDA
imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((224,224))).cuda() for img_path in paths])

# Move your model to GPU
model.cuda()

#adapt maybe if we take more images to not run out of memory
with torch.no_grad():
    # Ensure the input tensor is on GPU by calling .cuda() on it
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

features = features.reshape(len(paths) * 256, feat_dim).cpu().numpy()

pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

images_for_plotting = [Image.open(img_path).convert('RGB').resize((224, 224)) for img_path in paths[:25]]

fig, axs = plt.subplots(5, 5)
for i, ax in enumerate(axs.flat):
    if(i<min(25,number_of_images)):
        ax.imshow(images_for_plotting[i])
        ax.axis('off')  # Remove axis
plt.savefig("images_model2_dataset2.png")
plt.close() 

#otsu
first_component = pca_features[:, 0]

# Use Otsu's method to find an optimal threshold
threshold = threshold_otsu(first_component)

print("otsu optimal threshold:", threshold)

# segment using the first component
pca_features_bg = pca_features[:, 0] < threshold
pca_features_fg = ~pca_features_bg

# plot the pca_features_bg
for i in range(min(25,number_of_images)):
    plt.subplot(5, 5, i+1)
    plt.imshow(pca_features_bg[i * 256: (i+1) * 256].reshape(16, 16))
    plt.axis('off')
plt.savefig("pca_features_bg_model2_dataset2.png")
plt.close()

# PCA for only foreground patches
pca_features_rem = pca.transform(features[pca_features_fg])
for i in range(3):
    pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
    # transform using mean and std, I personally found this transformation gives a better visualization
    #pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

pca_features_rgb = pca_features.copy()
pca_features_rgb[pca_features_bg] = 0
pca_features_rgb[pca_features_fg] = pca_features_rem

pca_features_rgb = pca_features_rgb.reshape(number_of_images, patch_h, patch_w, 3)

for i in range(min(25,number_of_images)):
    plt.subplot(5, 5, i+1)
    plt.imshow(pca_features_rgb[i])
    plt.axis('off')
plt.savefig('pca_features_vis_model2_dataset2.png')
plt.close() 

for i in range(min(25,number_of_images)):
    plt.subplot(5, 5, i+1)
    image_array = np.array(pca_features_rgb[i])
    dominant_channel = np.argmax(image_array, axis=-1)

    segmented_image = np.zeros((*image_array.shape[:2], 3), dtype=np.uint8)

    non_black_mask = np.any(image_array != [0, 0, 0], axis=-1)

    segmented_image[(dominant_channel == 0) & non_black_mask] = [255,0,0]
    segmented_image[dominant_channel == 1] = [0,255,0]
    segmented_image[dominant_channel == 2] = [0,0,255]


    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
plt.savefig('features_segmented_model2_dataset2.png')
plt.close()