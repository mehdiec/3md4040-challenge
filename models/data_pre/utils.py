import os
import csv
from PIL import Image
import numpy as np


DIM = 28, 28


COL = ["img_class", "img_name"] + [
    f"part_{i}_x part_{j}_y" for i in range(28) for j in range(28)
]


def transform_images_from_folder(train_dir, new_file="data/train/train_csv/train.csv"):
    images = []
    with open(new_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(COL)
    for folder in train_dir:

        classe = int(folder[:3])
        real_folder = "data/train/" + folder
        for filename in os.listdir(real_folder):
            img_file = Image.open(os.path.join(real_folder, filename))
            if img_file is not None:

                img_file = img_file.resize(DIM)

                # Make image Greyscale
                img_grey = img_file.convert("L")
                # img_grey.save('result.png')
                # img_grey.show()

                # Save Greyscale values
                value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(
                    (img_grey.size[1], img_grey.size[0])
                )

                value = value.flatten()
                value = np.append([folder + filename, classe], value)

                with open(new_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(value)
    return images


def transform_images_from_test(new_file="data/test/test_csv/test.csv"):
    images = []
    with open(new_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(COL)

    classe = 420
    real_folder = "data/test/"
    for filename in os.listdir(real_folder):
        img_file = Image.open(os.path.join(real_folder, filename))
        if img_file is not None:

            img_file = img_file.resize(DIM)

            # Make image Greyscale
            img_grey = img_file.convert("L")
            # img_grey.save('result.png')
            # img_grey.show()

            # Save Greyscale values
            value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(
                (img_grey.size[1], img_grey.size[0])
            )

            value = value.flatten()
            value = np.append([filename, classe], value)

            with open(new_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(value)
    return images
