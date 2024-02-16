import os
import shutil
from PIL import Image

def tile_image(image_path, output_dir, tile_size=(512, 512)):
    """
    Divide an image into tiles of the specified size and save them to the output directory.
    Adjust the last tile to ensure it covers the end of the image, even if it overlaps.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        x_tiles = img_width // tile_size[0]
        y_tiles = img_height // tile_size[1]
        last_x = img_width - tile_size[0]
        last_y = img_height - tile_size[1]

        for x in range(0, img_width, tile_size[0]):
            for y in range(0, img_height, tile_size[1]):
                if x + tile_size[0] > img_width:
                    x = last_x
                if y + tile_size[1] > img_height:
                    y = last_y
                cropped_img = img.crop((x, y, x + tile_size[0], y + tile_size[1]))
                cropped_img.save(os.path.join(output_dir, f"{base_name}_{x}_{y}.png"))

def main():
    output_dir = "processed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir('.'):
        if file.endswith('.png'):
            file_path = os.path.join('.', file)
            if os.path.getsize(file_path) > 1 * 1024 * 1024:  # Greater than 1MB
                tile_image(file_path, output_dir)
            else:  # For images less than 1MB, copy them to the new directory
                shutil.copy(file_path, os.path.join(output_dir, file))

if __name__ == "__main__":
    main()
