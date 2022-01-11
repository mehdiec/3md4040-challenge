import os
import os.path
import argparse
import numpy as np
from PIL import Image
import PIL
import torchvision.transforms.functional as F

 ############################################################################################ Datasets
TRAIN_CONST = [
    "000_Candaciidae",
    "043_larvae__Annelida",
    "001_detritus",
    "044_Rhopalonema",
    "002_Calocalanus_pavo",
    "045_egg__other",
    "003_larvae__Crustacea",
    "046_tail__Appendicularia",
    "004_Podon",
    "047_Euchirella",
    "005_Sapphirinidae",
    "048_calyptopsis",
    "006_Calanidae",
    "049_Haloptilus",
    "007_zoea__Decapoda",
    "050_eudoxie__Diphyidae",
    "008_Gammaridea",
    "051_egg__Actinopterygii",
    "009_Oikopleuridae",
    "052_nectophore__Diphyidae",
    "010_Hyperiidea",
    "053_head",
    "011_zoea__Galatheidae",
    "054_Penilia",
    "012_nectophore__Physonectae",
    "055_egg__Cavolinia_inflexa",
    "013_Rhincalanidae",
    "056_Pontellidae",
    "014_Acantharea",
    "057_Coscinodiscus",
    "015_Foraminifera",
    "058_Acartiidae",
    "016_nauplii__Crustacea",
    "059_Corycaeidae",
    "017_gonophore__Diphyidae",
    "060_artefact",
    "018_metanauplii",
    "061_cirrus",
    "019_megalopa",
    "062_Luciferidae",
    "020_Brachyura",
    "063_Limacinidae",
    "021_tail__Chaetognatha",
    "064_cyphonaute",
    "022_Doliolida",
    "065_part__Copepoda",
    "023_Scyphozoa",
    "066_Fritillariidae",
    "024_Ctenophora",
    "067_Echinoidea",
    "025_Bivalvia__Mollusca",
    "068_Neoceratium",
    "026_ephyra",
    "069_Phaeodaria",
    "027_Temoridae",
    "070_Ostracoda",
    "028_scale",
    "071_Centropagidae",
    "029_Evadne",
    "072_Ophiuroidea",
    "030_Copilia",
    "073_nauplii__Cirripedia",
    "031_Eucalanidae",
    "074_Salpida",
    "032_Pyrosomatida",
    "075_Oithonidae",
    "033_nectophore__Abylopsis_tetragona",
    "076_eudoxie__Abylopsis_tetragona",
    "034_Actinopterygii",
    "077_cypris",
    "035_Creseidae",
    "078_Oncaeidae",
    "036_Calanoida",
    "079_gonophore__Abylopsis_tetragona",
    "037_Decapoda",
    "080_Harpacticoida",
    "038_Obelia",
    "081_Cavoliniidae",
    "039_Noctiluca",
    "082_Aglaura",
    "040_Spumellaria",
    "083_Euchaetidae",
    "041_Chaetognatha",
    "084_Tomopteridae",
    "042_Annelida",
    "085_Limacidae",
]

train_dataset_directory = "/usr/users/gpusdi1/gpusdi1_10/Téléchargements/train/"
new_train_dir1 = "/usr/users/gpusdi1/gpusdi1_10/Téléchargements/new_train/"


def square(dir_train,new_train_dir,classes_list=TRAIN_CONST):
    for elt in classes_list :
        
        dir_class = os.path.join(dir_train,elt)
        new_elt_dir = os.path.join(new_train_dir,elt)
        for im in os.listdir(dir_class):
            dir_image=os.path.join(dir_class,im)
            new_im_dir = os.path.join(new_elt_dir,im)
            image = Image.open(dir_image)
            max_wh = 300  # Max longueur largeur des images du dataset ������ determiner
            p_left, p_top = [(max_wh - s) // 2 for s in image.size]
            p_right, p_bottom = [
                max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
            ]
            padding = (p_left, p_top, p_right, p_bottom)
            #convert_tensor = transforms.ToTensor()
            #image = convert_tensor(image)
            #print(image)
            new_image = F.pad(image, padding, 255, "constant")  # valeur 0 pour la couleur noir, 255 pour blanche
            print(elt[:3])
            new_image = new_image.save(new_im_dir)
            break


'''
images_list = [image]
image = np.array(image)

for i in range(15):
    augmentations = transform(image=image)
    augmented_img = augmentations["image"]
    images_list.append(augmented_img)
plot_examples(images_list)

'''
square(train_dataset_directory,new_train_dir1)




        


