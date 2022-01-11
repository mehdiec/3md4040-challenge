import os
import os.path
import argparse
import numpy as np
from PIL import Image
import albumentations as A
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


def img_size(dir_train,classes_list=TRAIN_CONST):
    L={}
    for elt in classes_list :
        dir_class = os.path.join(dir_train,elt)
        l = len(os.listdir(dir_class))
        d = int((900/710 -1)*l)
        L[elt]=d
    return L


dict_ajout=img_size("/usr/users/gpusdi1/gpusdi1_10/Téléchargements/train/")
for k,v in dict_ajout.items():
    print("\"{}\" : {}".format(k,v))

dict_ajout = {  "000_Candaciidae" : 530
                "043_larvae__Annelida" : 45
                "001_detritus" : 0
                "044_Rhopalonema" : 82
                "002_Calocalanus_pavo" : 57
                "045_egg__other" : 3960
                "003_larvae__Crustacea" : 170
                "046_tail__Appendicularia" : 1233
                "004_Podon" : 354
                "047_Euchirella" : 41
                "005_Sapphirinidae" : 355
                "048_calyptopsis" : 1043
                "006_Calanidae" : 6368
                "049_Haloptilus" : 1065
                "007_zoea__Decapoda" : 1547
                "050_eudoxie__Diphyidae" : 832
                "008_Gammaridea" : 77
                "051_egg__Actinopterygii" : 532
                "009_Oikopleuridae" : 4939
                "052_nectophore__Diphyidae" : 1359
                "010_Hyperiidea" : 143
                "053_head" : 65
                "011_zoea__Galatheidae" : 49
                "054_Penilia" : 1704
                "012_nectophore__Physonectae" : 279
                "055_egg__Cavolinia_inflexa" : 202
                "013_Rhincalanidae" : 323
                "056_Pontellidae" : 344
                "014_Acantharea" : 183
                "057_Coscinodiscus" : 3381
                "015_Foraminifera" : 1490
                "058_Acartiidae" : 11265
                "016_nauplii__Crustacea" : 2248
                "059_Corycaeidae" : 2678
                "017_gonophore__Diphyidae" : 818
                "060_artefact" : 927
                "018_metanauplii" : 44
                "061_cirrus" : 71
                "019_megalopa" : 255
                "062_Luciferidae" : 122
                "020_Brachyura" : 1274
                "063_Limacinidae" : 2686
                "021_tail__Chaetognatha" : 755
                "064_cyphonaute" : 2197
                "022_Doliolida" : 867
                "065_part__Copepoda" : 99
                "023_Scyphozoa" : 20
                "066_Fritillariidae" : 619
                "024_Ctenophora" : 9
                "067_Echinoidea" : 62
                "025_Bivalvia__Mollusca" : 1047
                "068_Neoceratium" : 157
                "026_ephyra" : 26
                "069_Phaeodaria" : 4129
                "027_Temoridae" : 2194
                "070_Ostracoda" : 3466
                "028_scale" : 154
                "071_Centropagidae" : 1175
                "029_Evadne" : 5144
                "072_Ophiuroidea" : 343
                "030_Copilia" : 183
                "073_nauplii__Cirripedia" : 1499
                "031_Eucalanidae" : 1127
                "074_Salpida" : 1872
                "032_Pyrosomatida" : 89
                "075_Oithonidae" : 14286
                "033_nectophore__Abylopsis_tetragona" : 50
                "076_eudoxie__Abylopsis_tetragona" : 62
                "034_Actinopterygii" : 390
                "077_cypris" : 169
                "035_Creseidae" : 500
                "078_Oncaeidae" : 3074
                "036_Calanoida" : 48169
                "079_gonophore__Abylopsis_tetragona" : 23
                "037_Decapoda" : 946
                "080_Harpacticoida" : 692
                "038_Obelia" : 291
                "081_Cavoliniidae" : 673
                "039_Noctiluca" : 1620
                "082_Aglaura" : 103
                "040_Spumellaria" : 38
                "083_Euchaetidae" : 558
                "041_Chaetognatha" : 8870
                "084_Tomopteridae" : 46
                "042_Annelida" : 557
                "085_Limacidae" : 194
}


## Data Augmentation
def img_size(dir_train,classes_list=TRAIN_CONST):

    for elt in classes_list : 
        infos[elt]={}
        dir_class = os.path.join(dir_train,elt)
        for image in os.listdir(dir_class):
            dir_image=os.path.join(dir_class,image)
            im = Image.open(dir_image)
            width,height = im.size
            infos[elt][image]=(width,height)
    return infos
images_list = [image]
image = np.array(image)

for i in range(15):
    augmentations = transform(image=image)
    augmented_img = augmentations["image"]
    images_list.append(augmented_img)
plot_examples(images_list)