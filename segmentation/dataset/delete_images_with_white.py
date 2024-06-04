from PIL import Image
import os


def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = str(os.path.join(root, filename))
            file_paths.append(filepath)

    # path/124.tiff, path/123.tiff, ... - path/123.tiff, path/124.tiff, ...
    return sorted(file_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))


save_path = "/home/extio1/Desktop/image_class_data"
image_paths = get_all_file_paths(save_path + "/test_sample")
mask_paths = get_all_file_paths(save_path + "/test_target")

for i in range(len(image_paths)):
    image_path = image_paths[i]

    image = Image.open(image_path)
    total_pixels = image.width * image.height

    white_pixels = sum(
        1 for pixel in image.getdata() if pixel == (255, 255, 255))

    white_percentage = (white_pixels / total_pixels) * 100

    if white_percentage > 20:
        os.remove(image_paths[i])
        os.remove(mask_paths[i])
