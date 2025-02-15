# %%
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from torchinfo import summary
from torchmetrics import Accuracy, ConfusionMatrix

# %%
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# %%
def load_testing_data(src_dir):
    gallery_images=[]
    gallery_infos=[]
    
    probe_images=[]
    probe_infos=[]
    
    id = ["%03d" % i for i in range(75, 125)]
#gallery
    categories = ["nm-01", "nm-02", "nm-03", "nm-04"]
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    input_image = Image.open(path).convert("RGB")
                    input_tensor = preprocess(input_image)
                    #input_tensor=input_tensor.reshape((1,3,224,224))
                    gallery_images.append(input_tensor)
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    view="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    gallery_infos.append((label, view))
                    
#probe

    #categories = ["nm-05", "nm-06"]
    # categories = ["bg-01", "bg-02"]
    categories = ["cl-01", "cl-02"]
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    input_image = Image.open(path).convert("RGB")

                    if np.array(input_image).sum() == 0:
                         input_tensor = torch.empty(0)
                    else:    
                        input_tensor = preprocess(input_image)
                        
                    probe_images.append(input_tensor)
                    label = "{0:03}".format(int(l.split("-")[0])-1)
                    view="{0:03}".format(int(l.split("-")[3].split(".")[0]))
                    probe_infos.append((label, view))
                    
    return gallery_images, gallery_infos,probe_images,probe_infos


# gallery_images_head, gallery_infos_head, probe_images_head, probe_infos_head = load_testing_data('D:/5Part_Gei/1_PART')
# gallery_images_chest1, gallery_infos_chest1, probe_images_chest1, probe_infos_chest1 = load_testing_data("D:/5Part_Gei/2_PART")
# gallery_images_chest2, gallery_infos_chest2, probe_images_chest2, probe_infos_chest2 = load_testing_data("D:/5Part_Gei/3_PART")
# gallery_images_foot, gallery_infos_foot, probe_images_foot, probe_infos_foot = load_testing_data("D:/5Part_Gei/4_PART")
# gallery_images_foot2, gallery_infos_foot2, probe_images_foot2, probe_infos_foot2 = load_testing_data("D:/5Part_Gei/5_PART")

gallery_images_head, gallery_infos_head, probe_images_head, probe_infos_head = load_testing_data('D:/CASIA_WITHOUT_VARIATION/1_PART')
gallery_images_chest1, gallery_infos_chest1, probe_images_chest1, probe_infos_chest1 = load_testing_data("D:/CASIA_WITHOUT_VARIATION/2_PART")
gallery_images_chest2, gallery_infos_chest2, probe_images_chest2, probe_infos_chest2 = load_testing_data("D:/CASIA_WITHOUT_VARIATION/3_PART")
gallery_images_foot, gallery_infos_foot, probe_images_foot, probe_infos_foot = load_testing_data("D:/CASIA_WITHOUT_VARIATION/4_PART")
gallery_images_foot2, gallery_infos_foot2, probe_images_foot2, probe_infos_foot2 = load_testing_data("D:/CASIA_WITHOUT_VARIATION/5_PART")


# %%
gallery_000_images_head=[]
gallery_000_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='000':
        gallery_000_images_head.append(gallery_images_head[i])
        gallery_000_infos_head.append(gallery_infos_head[i])

gallery_018_images_head=[]
gallery_018_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='018':
        gallery_018_images_head.append(gallery_images_head[i])
        gallery_018_infos_head.append(gallery_infos_head[i])

gallery_036_images_head=[]
gallery_036_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='036':
        gallery_036_images_head.append(gallery_images_head[i])
        gallery_036_infos_head.append(gallery_infos_head[i])

gallery_054_images_head=[]
gallery_054_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='054':
        gallery_054_images_head.append(gallery_images_head[i])
        gallery_054_infos_head.append(gallery_infos_head[i])

gallery_072_images_head=[]
gallery_072_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='072':
        gallery_072_images_head.append(gallery_images_head[i])
        gallery_072_infos_head.append(gallery_infos_head[i])

gallery_090_images_head=[]
gallery_090_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='090':
        gallery_090_images_head.append(gallery_images_head[i])
        gallery_090_infos_head.append(gallery_infos_head[i])

gallery_108_images_head=[]
gallery_108_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='108':
        gallery_108_images_head.append(gallery_images_head[i])
        gallery_108_infos_head.append(gallery_infos_head[i])

gallery_126_images_head=[]
gallery_126_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='126':
        gallery_126_images_head.append(gallery_images_head[i])
        gallery_126_infos_head.append(gallery_infos_head[i])

gallery_144_images_head=[]
gallery_144_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='144':
        gallery_144_images_head.append(gallery_images_head[i])
        gallery_144_infos_head.append(gallery_infos_head[i])

gallery_162_images_head=[]
gallery_162_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='162':
        gallery_162_images_head.append(gallery_images_head[i])
        gallery_162_infos_head.append(gallery_infos_head[i])

gallery_180_images_head=[]
gallery_180_infos_head=[]

for i in range(len(gallery_images_head)):
    if gallery_infos_head[i][1]=='180':
        gallery_180_images_head.append(gallery_images_head[i])
        gallery_180_infos_head.append(gallery_infos_head[i])


################################################################### PROBES STACK
probe_000_images_head=[]
probe_000_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='000':
        probe_000_images_head.append(probe_images_head[i])
        probe_000_infos_head.append(probe_infos_head[i])

probe_018_images_head=[]
probe_018_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='018':
        probe_018_images_head.append(probe_images_head[i])
        probe_018_infos_head.append(probe_infos_head[i])

probe_036_images_head=[]
probe_036_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='036':
        probe_036_images_head.append(probe_images_head[i])
        probe_036_infos_head.append(probe_infos_head[i])

probe_054_images_head=[]
probe_054_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='054':
        probe_054_images_head.append(probe_images_head[i])
        probe_054_infos_head.append(probe_infos_head[i])

probe_072_images_head=[]
probe_072_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='072':
        probe_072_images_head.append(probe_images_head[i])
        probe_072_infos_head.append(probe_infos_head[i])

probe_090_images_head=[]
probe_090_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='090':
        probe_090_images_head.append(probe_images_head[i])
        probe_090_infos_head.append(probe_infos_head[i])

probe_108_images_head=[]
probe_108_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='108':
        probe_108_images_head.append(probe_images_head[i])
        probe_108_infos_head.append(probe_infos_head[i])

probe_126_images_head=[]
probe_126_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='126':
        probe_126_images_head.append(probe_images_head[i])
        probe_126_infos_head.append(probe_infos_head[i])

probe_144_images_head=[]
probe_144_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='144':
        probe_144_images_head.append(probe_images_head[i])
        probe_144_infos_head.append(probe_infos_head[i])

probe_162_images_head=[]
probe_162_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='162':
        probe_162_images_head.append(probe_images_head[i])
        probe_162_infos_head.append(probe_infos_head[i])


probe_180_images_head=[]
probe_180_infos_head=[]

for i in range(len(probe_images_head)):
    if probe_infos_head[i][1]=='180':
        probe_180_images_head.append(probe_images_head[i])
        probe_180_infos_head.append(probe_infos_head[i])

######################################################################## ALL GALERIES STACK

All_gallery_angles_images_head=[]
All_gallery_angles_infos_head=[]

All_gallery_angles_images_head.append(gallery_000_images_head)
All_gallery_angles_infos_head.append(gallery_000_infos_head)

All_gallery_angles_images_head.append(gallery_018_images_head)
All_gallery_angles_infos_head.append(gallery_018_infos_head)

All_gallery_angles_images_head.append(gallery_036_images_head)
All_gallery_angles_infos_head.append(gallery_036_infos_head)

All_gallery_angles_images_head.append(gallery_054_images_head)
All_gallery_angles_infos_head.append(gallery_054_infos_head)

All_gallery_angles_images_head.append(gallery_072_images_head)
All_gallery_angles_infos_head.append(gallery_072_infos_head)

All_gallery_angles_images_head.append(gallery_090_images_head)
All_gallery_angles_infos_head.append(gallery_090_infos_head)

All_gallery_angles_images_head.append(gallery_108_images_head)
All_gallery_angles_infos_head.append(gallery_108_infos_head)

All_gallery_angles_images_head.append(gallery_126_images_head)
All_gallery_angles_infos_head.append(gallery_126_infos_head)

All_gallery_angles_images_head.append(gallery_144_images_head)
All_gallery_angles_infos_head.append(gallery_144_infos_head)

All_gallery_angles_images_head.append(gallery_162_images_head)
All_gallery_angles_infos_head.append(gallery_162_infos_head)

All_gallery_angles_images_head.append(gallery_180_images_head)
All_gallery_angles_infos_head.append(gallery_180_infos_head)
########################################################################### ALL PROBES STACK

All_probe_angles_images_head=[]
All_probe_angles_infos_head=[]

All_probe_angles_images_head.append(probe_000_images_head)
All_probe_angles_infos_head.append(probe_000_infos_head)

All_probe_angles_images_head.append(probe_018_images_head)
All_probe_angles_infos_head.append(probe_018_infos_head)

All_probe_angles_images_head.append(probe_036_images_head)
All_probe_angles_infos_head.append(probe_036_infos_head)

All_probe_angles_images_head.append(probe_054_images_head)
All_probe_angles_infos_head.append(probe_054_infos_head)

All_probe_angles_images_head.append(probe_072_images_head)
All_probe_angles_infos_head.append(probe_072_infos_head)

All_probe_angles_images_head.append(probe_090_images_head)
All_probe_angles_infos_head.append(probe_090_infos_head)

All_probe_angles_images_head.append(probe_108_images_head)
All_probe_angles_infos_head.append(probe_108_infos_head)

All_probe_angles_images_head.append(probe_126_images_head)
All_probe_angles_infos_head.append(probe_126_infos_head)

All_probe_angles_images_head.append(probe_144_images_head)
All_probe_angles_infos_head.append(probe_144_infos_head)

All_probe_angles_images_head.append(probe_162_images_head)
All_probe_angles_infos_head.append(probe_162_infos_head)

All_probe_angles_images_head.append(probe_180_images_head)
All_probe_angles_infos_head.append(probe_180_infos_head)

# %%
gallery_000_images_foot=[]
gallery_000_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='000':
        gallery_000_images_foot.append(gallery_images_foot[i])
        gallery_000_infos_foot.append(gallery_infos_foot[i])

gallery_018_images_foot=[]
gallery_018_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='018':
        gallery_018_images_foot.append(gallery_images_foot[i])
        gallery_018_infos_foot.append(gallery_infos_foot[i])

gallery_036_images_foot=[]
gallery_036_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='036':
        gallery_036_images_foot.append(gallery_images_foot[i])
        gallery_036_infos_foot.append(gallery_infos_foot[i])

gallery_054_images_foot=[]
gallery_054_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='054':
        gallery_054_images_foot.append(gallery_images_foot[i])
        gallery_054_infos_foot.append(gallery_infos_foot[i])

gallery_072_images_foot=[]
gallery_072_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='072':
        gallery_072_images_foot.append(gallery_images_foot[i])
        gallery_072_infos_foot.append(gallery_infos_foot[i])

gallery_090_images_foot=[]
gallery_090_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='090':
        gallery_090_images_foot.append(gallery_images_foot[i])
        gallery_090_infos_foot.append(gallery_infos_foot[i])

gallery_108_images_foot=[]
gallery_108_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='108':
        gallery_108_images_foot.append(gallery_images_foot[i])
        gallery_108_infos_foot.append(gallery_infos_foot[i])

gallery_126_images_foot=[]
gallery_126_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='126':
        gallery_126_images_foot.append(gallery_images_foot[i])
        gallery_126_infos_foot.append(gallery_infos_foot[i])

gallery_144_images_foot=[]
gallery_144_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='144':
        gallery_144_images_foot.append(gallery_images_foot[i])
        gallery_144_infos_foot.append(gallery_infos_foot[i])

gallery_162_images_foot=[]
gallery_162_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='162':
        gallery_162_images_foot.append(gallery_images_foot[i])
        gallery_162_infos_foot.append(gallery_infos_foot[i])

gallery_180_images_foot=[]
gallery_180_infos_foot=[]

for i in range(len(gallery_images_foot)):
    if gallery_infos_foot[i][1]=='180':
        gallery_180_images_foot.append(gallery_images_foot[i])
        gallery_180_infos_foot.append(gallery_infos_foot[i])


################################################################### PROBES GEI
probe_000_images_foot=[]
probe_000_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='000':
        probe_000_images_foot.append(probe_images_foot[i])
        probe_000_infos_foot.append(probe_infos_foot[i])

probe_018_images_foot=[]
probe_018_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='018':
        probe_018_images_foot.append(probe_images_foot[i])
        probe_018_infos_foot.append(probe_infos_foot[i])

probe_036_images_foot=[]
probe_036_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='036':
        probe_036_images_foot.append(probe_images_foot[i])
        probe_036_infos_foot.append(probe_infos_foot[i])

probe_054_images_foot=[]
probe_054_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='054':
        probe_054_images_foot.append(probe_images_foot[i])
        probe_054_infos_foot.append(probe_infos_foot[i])

probe_072_images_foot=[]
probe_072_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='072':
        probe_072_images_foot.append(probe_images_foot[i])
        probe_072_infos_foot.append(probe_infos_foot[i])

probe_090_images_foot=[]
probe_090_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='090':
        probe_090_images_foot.append(probe_images_foot[i])
        probe_090_infos_foot.append(probe_infos_foot[i])

probe_108_images_foot=[]
probe_108_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='108':
        probe_108_images_foot.append(probe_images_foot[i])
        probe_108_infos_foot.append(probe_infos_foot[i])

probe_126_images_foot=[]
probe_126_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='126':
        probe_126_images_foot.append(probe_images_foot[i])
        probe_126_infos_foot.append(probe_infos_foot[i])

probe_144_images_foot=[]
probe_144_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='144':
        probe_144_images_foot.append(probe_images_foot[i])
        probe_144_infos_foot.append(probe_infos_foot[i])

probe_162_images_foot=[]
probe_162_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='162':
        probe_162_images_foot.append(probe_images_foot[i])
        probe_162_infos_foot.append(probe_infos_foot[i])


probe_180_images_foot=[]
probe_180_infos_foot=[]

for i in range(len(probe_images_foot)):
    if probe_infos_foot[i][1]=='180':
        probe_180_images_foot.append(probe_images_foot[i])
        probe_180_infos_foot.append(probe_infos_foot[i])

######################################################################## ALL GALERIES GEI

All_gallery_angles_images_foot=[]
All_gallery_angles_infos_foot=[]

All_gallery_angles_images_foot.append(gallery_000_images_foot)
All_gallery_angles_infos_foot.append(gallery_000_infos_foot)

All_gallery_angles_images_foot.append(gallery_018_images_foot)
All_gallery_angles_infos_foot.append(gallery_018_infos_foot)

All_gallery_angles_images_foot.append(gallery_036_images_foot)
All_gallery_angles_infos_foot.append(gallery_036_infos_foot)

All_gallery_angles_images_foot.append(gallery_054_images_foot)
All_gallery_angles_infos_foot.append(gallery_054_infos_foot)

All_gallery_angles_images_foot.append(gallery_072_images_foot)
All_gallery_angles_infos_foot.append(gallery_072_infos_foot)

All_gallery_angles_images_foot.append(gallery_090_images_foot)
All_gallery_angles_infos_foot.append(gallery_090_infos_foot)

All_gallery_angles_images_foot.append(gallery_108_images_foot)
All_gallery_angles_infos_foot.append(gallery_108_infos_foot)

All_gallery_angles_images_foot.append(gallery_126_images_foot)
All_gallery_angles_infos_foot.append(gallery_126_infos_foot)

All_gallery_angles_images_foot.append(gallery_144_images_foot)
All_gallery_angles_infos_foot.append(gallery_144_infos_foot)

All_gallery_angles_images_foot.append(gallery_162_images_foot)
All_gallery_angles_infos_foot.append(gallery_162_infos_foot)

All_gallery_angles_images_foot.append(gallery_180_images_foot)
All_gallery_angles_infos_foot.append(gallery_180_infos_foot)
########################################################################### ALL PROBES GEI

All_probe_angles_images_foot=[]
All_probe_angles_infos_foot=[]

All_probe_angles_images_foot.append(probe_000_images_foot)
All_probe_angles_infos_foot.append(probe_000_infos_foot)

All_probe_angles_images_foot.append(probe_018_images_foot)
All_probe_angles_infos_foot.append(probe_018_infos_foot)

All_probe_angles_images_foot.append(probe_036_images_foot)
All_probe_angles_infos_foot.append(probe_036_infos_foot)

All_probe_angles_images_foot.append(probe_054_images_foot)
All_probe_angles_infos_foot.append(probe_054_infos_foot)

All_probe_angles_images_foot.append(probe_072_images_foot)
All_probe_angles_infos_foot.append(probe_072_infos_foot)

All_probe_angles_images_foot.append(probe_090_images_foot)
All_probe_angles_infos_foot.append(probe_090_infos_foot)

All_probe_angles_images_foot.append(probe_108_images_foot)
All_probe_angles_infos_foot.append(probe_108_infos_foot)

All_probe_angles_images_foot.append(probe_126_images_foot)
All_probe_angles_infos_foot.append(probe_126_infos_foot)

All_probe_angles_images_foot.append(probe_144_images_foot)
All_probe_angles_infos_foot.append(probe_144_infos_foot)

All_probe_angles_images_foot.append(probe_162_images_foot)
All_probe_angles_infos_foot.append(probe_162_infos_foot)

All_probe_angles_images_foot.append(probe_180_images_foot)
All_probe_angles_infos_foot.append(probe_180_infos_foot)



# %%
gallery_000_images_chest1=[]
gallery_000_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='000':
        gallery_000_images_chest1.append(gallery_images_chest1[i])
        gallery_000_infos_chest1.append(gallery_infos_chest1[i])

gallery_018_images_chest1=[]
gallery_018_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='018':
        gallery_018_images_chest1.append(gallery_images_chest1[i])
        gallery_018_infos_chest1.append(gallery_infos_chest1[i])

gallery_036_images_chest1=[]
gallery_036_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='036':
        gallery_036_images_chest1.append(gallery_images_chest1[i])
        gallery_036_infos_chest1.append(gallery_infos_chest1[i])

gallery_054_images_chest1=[]
gallery_054_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='054':
        gallery_054_images_chest1.append(gallery_images_chest1[i])
        gallery_054_infos_chest1.append(gallery_infos_chest1[i])

gallery_072_images_chest1=[]
gallery_072_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='072':
        gallery_072_images_chest1.append(gallery_images_chest1[i])
        gallery_072_infos_chest1.append(gallery_infos_chest1[i])

gallery_090_images_chest1=[]
gallery_090_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='090':
        gallery_090_images_chest1.append(gallery_images_chest1[i])
        gallery_090_infos_chest1.append(gallery_infos_chest1[i])

gallery_108_images_chest1=[]
gallery_108_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='108':
        gallery_108_images_chest1.append(gallery_images_chest1[i])
        gallery_108_infos_chest1.append(gallery_infos_chest1[i])

gallery_126_images_chest1=[]
gallery_126_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='126':
        gallery_126_images_chest1.append(gallery_images_chest1[i])
        gallery_126_infos_chest1.append(gallery_infos_chest1[i])

gallery_144_images_chest1=[]
gallery_144_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='144':
        gallery_144_images_chest1.append(gallery_images_chest1[i])
        gallery_144_infos_chest1.append(gallery_infos_chest1[i])

gallery_162_images_chest1=[]
gallery_162_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='162':
        gallery_162_images_chest1.append(gallery_images_chest1[i])
        gallery_162_infos_chest1.append(gallery_infos_chest1[i])

gallery_180_images_chest1=[]
gallery_180_infos_chest1=[]

for i in range(len(gallery_images_chest1)):
    if gallery_infos_chest1[i][1]=='180':
        gallery_180_images_chest1.append(gallery_images_chest1[i])
        gallery_180_infos_chest1.append(gallery_infos_chest1[i])


################################################################### PROBES STACK
probe_000_images_chest1=[]
probe_000_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='000':
        probe_000_images_chest1.append(probe_images_chest1[i])
        probe_000_infos_chest1.append(probe_infos_chest1[i])

probe_018_images_chest1=[]
probe_018_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='018':
        probe_018_images_chest1.append(probe_images_chest1[i])
        probe_018_infos_chest1.append(probe_infos_chest1[i])

probe_036_images_chest1=[]
probe_036_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='036':
        probe_036_images_chest1.append(probe_images_chest1[i])
        probe_036_infos_chest1.append(probe_infos_chest1[i])

probe_054_images_chest1=[]
probe_054_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='054':
        probe_054_images_chest1.append(probe_images_chest1[i])
        probe_054_infos_chest1.append(probe_infos_chest1[i])

probe_072_images_chest1=[]
probe_072_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='072':
        probe_072_images_chest1.append(probe_images_chest1[i])
        probe_072_infos_chest1.append(probe_infos_chest1[i])

probe_090_images_chest1=[]
probe_090_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='090':
        probe_090_images_chest1.append(probe_images_chest1[i])
        probe_090_infos_chest1.append(probe_infos_chest1[i])

probe_108_images_chest1=[]
probe_108_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='108':
        probe_108_images_chest1.append(probe_images_chest1[i])
        probe_108_infos_chest1.append(probe_infos_chest1[i])

probe_126_images_chest1=[]
probe_126_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='126':
        probe_126_images_chest1.append(probe_images_chest1[i])
        probe_126_infos_chest1.append(probe_infos_chest1[i])

probe_144_images_chest1=[]
probe_144_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='144':
        probe_144_images_chest1.append(probe_images_chest1[i])
        probe_144_infos_chest1.append(probe_infos_chest1[i])

probe_162_images_chest1=[]
probe_162_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='162':
        probe_162_images_chest1.append(probe_images_chest1[i])
        probe_162_infos_chest1.append(probe_infos_chest1[i])


probe_180_images_chest1=[]
probe_180_infos_chest1=[]

for i in range(len(probe_images_chest1)):
    if probe_infos_chest1[i][1]=='180':
        probe_180_images_chest1.append(probe_images_chest1[i])
        probe_180_infos_chest1.append(probe_infos_chest1[i])

######################################################################## ALL GALERIES STACK

All_gallery_angles_images_chest1=[]
All_gallery_angles_infos_chest1=[]

All_gallery_angles_images_chest1.append(gallery_000_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_000_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_018_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_018_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_036_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_036_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_054_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_054_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_072_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_072_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_090_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_090_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_108_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_108_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_126_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_126_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_144_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_144_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_162_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_162_infos_chest1)

All_gallery_angles_images_chest1.append(gallery_180_images_chest1)
All_gallery_angles_infos_chest1.append(gallery_180_infos_chest1)
########################################################################### ALL PROBES STACK

All_probe_angles_images_chest1=[]
All_probe_angles_infos_chest1=[]

All_probe_angles_images_chest1.append(probe_000_images_chest1)
All_probe_angles_infos_chest1.append(probe_000_infos_chest1)

All_probe_angles_images_chest1.append(probe_018_images_chest1)
All_probe_angles_infos_chest1.append(probe_018_infos_chest1)

All_probe_angles_images_chest1.append(probe_036_images_chest1)
All_probe_angles_infos_chest1.append(probe_036_infos_chest1)

All_probe_angles_images_chest1.append(probe_054_images_chest1)
All_probe_angles_infos_chest1.append(probe_054_infos_chest1)

All_probe_angles_images_chest1.append(probe_072_images_chest1)
All_probe_angles_infos_chest1.append(probe_072_infos_chest1)

All_probe_angles_images_chest1.append(probe_090_images_chest1)
All_probe_angles_infos_chest1.append(probe_090_infos_chest1)

All_probe_angles_images_chest1.append(probe_108_images_chest1)
All_probe_angles_infos_chest1.append(probe_108_infos_chest1)

All_probe_angles_images_chest1.append(probe_126_images_chest1)
All_probe_angles_infos_chest1.append(probe_126_infos_chest1)

All_probe_angles_images_chest1.append(probe_144_images_chest1)
All_probe_angles_infos_chest1.append(probe_144_infos_chest1)

All_probe_angles_images_chest1.append(probe_162_images_chest1)
All_probe_angles_infos_chest1.append(probe_162_infos_chest1)

All_probe_angles_images_chest1.append(probe_180_images_chest1)
All_probe_angles_infos_chest1.append(probe_180_infos_chest1)

# %%
gallery_000_images_chest2=[]
gallery_000_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='000':
        gallery_000_images_chest2.append(gallery_images_chest2[i])
        gallery_000_infos_chest2.append(gallery_infos_chest2[i])

gallery_018_images_chest2=[]
gallery_018_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='018':
        gallery_018_images_chest2.append(gallery_images_chest2[i])
        gallery_018_infos_chest2.append(gallery_infos_chest2[i])

gallery_036_images_chest2=[]
gallery_036_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='036':
        gallery_036_images_chest2.append(gallery_images_chest2[i])
        gallery_036_infos_chest2.append(gallery_infos_chest2[i])

gallery_054_images_chest2=[]
gallery_054_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='054':
        gallery_054_images_chest2.append(gallery_images_chest2[i])
        gallery_054_infos_chest2.append(gallery_infos_chest2[i])

gallery_072_images_chest2=[]
gallery_072_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='072':
        gallery_072_images_chest2.append(gallery_images_chest2[i])
        gallery_072_infos_chest2.append(gallery_infos_chest2[i])

gallery_090_images_chest2=[]
gallery_090_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='090':
        gallery_090_images_chest2.append(gallery_images_chest2[i])
        gallery_090_infos_chest2.append(gallery_infos_chest2[i])

gallery_108_images_chest2=[]
gallery_108_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='108':
        gallery_108_images_chest2.append(gallery_images_chest2[i])
        gallery_108_infos_chest2.append(gallery_infos_chest2[i])

gallery_126_images_chest2=[]
gallery_126_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='126':
        gallery_126_images_chest2.append(gallery_images_chest2[i])
        gallery_126_infos_chest2.append(gallery_infos_chest2[i])

gallery_144_images_chest2=[]
gallery_144_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='144':
        gallery_144_images_chest2.append(gallery_images_chest2[i])
        gallery_144_infos_chest2.append(gallery_infos_chest2[i])

gallery_162_images_chest2=[]
gallery_162_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='162':
        gallery_162_images_chest2.append(gallery_images_chest2[i])
        gallery_162_infos_chest2.append(gallery_infos_chest2[i])

gallery_180_images_chest2=[]
gallery_180_infos_chest2=[]

for i in range(len(gallery_images_chest2)):
    if gallery_infos_chest2[i][1]=='180':
        gallery_180_images_chest2.append(gallery_images_chest2[i])
        gallery_180_infos_chest2.append(gallery_infos_chest2[i])


################################################################### PROBES STACK
probe_000_images_chest2=[]
probe_000_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='000':
        probe_000_images_chest2.append(probe_images_chest2[i])
        probe_000_infos_chest2.append(probe_infos_chest2[i])

probe_018_images_chest2=[]
probe_018_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='018':
        probe_018_images_chest2.append(probe_images_chest2[i])
        probe_018_infos_chest2.append(probe_infos_chest2[i])

probe_036_images_chest2=[]
probe_036_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='036':
        probe_036_images_chest2.append(probe_images_chest2[i])
        probe_036_infos_chest2.append(probe_infos_chest2[i])

probe_054_images_chest2=[]
probe_054_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='054':
        probe_054_images_chest2.append(probe_images_chest2[i])
        probe_054_infos_chest2.append(probe_infos_chest2[i])

probe_072_images_chest2=[]
probe_072_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='072':
        probe_072_images_chest2.append(probe_images_chest2[i])
        probe_072_infos_chest2.append(probe_infos_chest2[i])

probe_090_images_chest2=[]
probe_090_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='090':
        probe_090_images_chest2.append(probe_images_chest2[i])
        probe_090_infos_chest2.append(probe_infos_chest2[i])

probe_108_images_chest2=[]
probe_108_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='108':
        probe_108_images_chest2.append(probe_images_chest2[i])
        probe_108_infos_chest2.append(probe_infos_chest2[i])

probe_126_images_chest2=[]
probe_126_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='126':
        probe_126_images_chest2.append(probe_images_chest2[i])
        probe_126_infos_chest2.append(probe_infos_chest2[i])

probe_144_images_chest2=[]
probe_144_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='144':
        probe_144_images_chest2.append(probe_images_chest2[i])
        probe_144_infos_chest2.append(probe_infos_chest2[i])

probe_162_images_chest2=[]
probe_162_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='162':
        probe_162_images_chest2.append(probe_images_chest2[i])
        probe_162_infos_chest2.append(probe_infos_chest2[i])


probe_180_images_chest2=[]
probe_180_infos_chest2=[]

for i in range(len(probe_images_chest2)):
    if probe_infos_chest2[i][1]=='180':
        probe_180_images_chest2.append(probe_images_chest2[i])
        probe_180_infos_chest2.append(probe_infos_chest2[i])

######################################################################## ALL GALERIES STACK

All_gallery_angles_images_chest2=[]
All_gallery_angles_infos_chest2=[]

All_gallery_angles_images_chest2.append(gallery_000_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_000_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_018_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_018_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_036_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_036_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_054_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_054_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_072_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_072_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_090_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_090_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_108_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_108_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_126_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_126_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_144_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_144_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_162_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_162_infos_chest2)

All_gallery_angles_images_chest2.append(gallery_180_images_chest2)
All_gallery_angles_infos_chest2.append(gallery_180_infos_chest2)
########################################################################### ALL PROBES STACK

All_probe_angles_images_chest2=[]
All_probe_angles_infos_chest2=[]

All_probe_angles_images_chest2.append(probe_000_images_chest2)
All_probe_angles_infos_chest2.append(probe_000_infos_chest2)

All_probe_angles_images_chest2.append(probe_018_images_chest2)
All_probe_angles_infos_chest2.append(probe_018_infos_chest2)

All_probe_angles_images_chest2.append(probe_036_images_chest2)
All_probe_angles_infos_chest2.append(probe_036_infos_chest2)

All_probe_angles_images_chest2.append(probe_054_images_chest2)
All_probe_angles_infos_chest2.append(probe_054_infos_chest2)

All_probe_angles_images_chest2.append(probe_072_images_chest2)
All_probe_angles_infos_chest2.append(probe_072_infos_chest2)

All_probe_angles_images_chest2.append(probe_090_images_chest2)
All_probe_angles_infos_chest2.append(probe_090_infos_chest2)

All_probe_angles_images_chest2.append(probe_108_images_chest2)
All_probe_angles_infos_chest2.append(probe_108_infos_chest2)

All_probe_angles_images_chest2.append(probe_126_images_chest2)
All_probe_angles_infos_chest2.append(probe_126_infos_chest2)

All_probe_angles_images_chest2.append(probe_144_images_chest2)
All_probe_angles_infos_chest2.append(probe_144_infos_chest2)

All_probe_angles_images_chest2.append(probe_162_images_chest2)
All_probe_angles_infos_chest2.append(probe_162_infos_chest2)

All_probe_angles_images_chest2.append(probe_180_images_chest2)
All_probe_angles_infos_chest2.append(probe_180_infos_chest2)

# %%
gallery_000_images_foot2=[]
gallery_000_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='000':
        gallery_000_images_foot2.append(gallery_images_foot2[i])
        gallery_000_infos_foot2.append(gallery_infos_foot2[i])

gallery_018_images_foot2=[]
gallery_018_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='018':
        gallery_018_images_foot2.append(gallery_images_foot2[i])
        gallery_018_infos_foot2.append(gallery_infos_foot2[i])

gallery_036_images_foot2=[]
gallery_036_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='036':
        gallery_036_images_foot2.append(gallery_images_foot2[i])
        gallery_036_infos_foot2.append(gallery_infos_foot2[i])

gallery_054_images_foot2=[]
gallery_054_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='054':
        gallery_054_images_foot2.append(gallery_images_foot2[i])
        gallery_054_infos_foot2.append(gallery_infos_foot2[i])

gallery_072_images_foot2=[]
gallery_072_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='072':
        gallery_072_images_foot2.append(gallery_images_foot2[i])
        gallery_072_infos_foot2.append(gallery_infos_foot2[i])

gallery_090_images_foot2=[]
gallery_090_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='090':
        gallery_090_images_foot2.append(gallery_images_foot2[i])
        gallery_090_infos_foot2.append(gallery_infos_foot2[i])

gallery_108_images_foot2=[]
gallery_108_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='108':
        gallery_108_images_foot2.append(gallery_images_foot2[i])
        gallery_108_infos_foot2.append(gallery_infos_foot2[i])

gallery_126_images_foot2=[]
gallery_126_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='126':
        gallery_126_images_foot2.append(gallery_images_foot2[i])
        gallery_126_infos_foot2.append(gallery_infos_foot2[i])

gallery_144_images_foot2=[]
gallery_144_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='144':
        gallery_144_images_foot2.append(gallery_images_foot2[i])
        gallery_144_infos_foot2.append(gallery_infos_foot2[i])

gallery_162_images_foot2=[]
gallery_162_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='162':
        gallery_162_images_foot2.append(gallery_images_foot2[i])
        gallery_162_infos_foot2.append(gallery_infos_foot2[i])

gallery_180_images_foot2=[]
gallery_180_infos_foot2=[]

for i in range(len(gallery_images_foot2)):
    if gallery_infos_foot2[i][1]=='180':
        gallery_180_images_foot2.append(gallery_images_foot2[i])
        gallery_180_infos_foot2.append(gallery_infos_foot2[i])


################################################################### PROBES GEI
probe_000_images_foot2=[]
probe_000_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='000':
        probe_000_images_foot2.append(probe_images_foot2[i])
        probe_000_infos_foot2.append(probe_infos_foot2[i])

probe_018_images_foot2=[]
probe_018_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='018':
        probe_018_images_foot2.append(probe_images_foot2[i])
        probe_018_infos_foot2.append(probe_infos_foot2[i])

probe_036_images_foot2=[]
probe_036_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='036':
        probe_036_images_foot2.append(probe_images_foot2[i])
        probe_036_infos_foot2.append(probe_infos_foot2[i])

probe_054_images_foot2=[]
probe_054_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='054':
        probe_054_images_foot2.append(probe_images_foot2[i])
        probe_054_infos_foot2.append(probe_infos_foot2[i])

probe_072_images_foot2=[]
probe_072_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='072':
        probe_072_images_foot2.append(probe_images_foot2[i])
        probe_072_infos_foot2.append(probe_infos_foot2[i])

probe_090_images_foot2=[]
probe_090_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='090':
        probe_090_images_foot2.append(probe_images_foot2[i])
        probe_090_infos_foot2.append(probe_infos_foot2[i])

probe_108_images_foot2=[]
probe_108_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='108':
        probe_108_images_foot2.append(probe_images_foot2[i])
        probe_108_infos_foot2.append(probe_infos_foot2[i])

probe_126_images_foot2=[]
probe_126_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='126':
        probe_126_images_foot2.append(probe_images_foot2[i])
        probe_126_infos_foot2.append(probe_infos_foot2[i])

probe_144_images_foot2=[]
probe_144_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='144':
        probe_144_images_foot2.append(probe_images_foot2[i])
        probe_144_infos_foot2.append(probe_infos_foot2[i])

probe_162_images_foot2=[]
probe_162_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='162':
        probe_162_images_foot2.append(probe_images_foot2[i])
        probe_162_infos_foot2.append(probe_infos_foot2[i])


probe_180_images_foot2=[]
probe_180_infos_foot2=[]

for i in range(len(probe_images_foot2)):
    if probe_infos_foot2[i][1]=='180':
        probe_180_images_foot2.append(probe_images_foot2[i])
        probe_180_infos_foot2.append(probe_infos_foot2[i])

######################################################################## ALL GALERIES GEI

All_gallery_angles_images_foot2=[]
All_gallery_angles_infos_foot2=[]

All_gallery_angles_images_foot2.append(gallery_000_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_000_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_018_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_018_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_036_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_036_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_054_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_054_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_072_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_072_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_090_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_090_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_108_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_108_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_126_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_126_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_144_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_144_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_162_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_162_infos_foot2)

All_gallery_angles_images_foot2.append(gallery_180_images_foot2)
All_gallery_angles_infos_foot2.append(gallery_180_infos_foot2)
########################################################################### ALL PROBES GEI

All_probe_angles_images_foot2=[]
All_probe_angles_infos_foot2=[]

All_probe_angles_images_foot2.append(probe_000_images_foot2)
All_probe_angles_infos_foot2.append(probe_000_infos_foot2)

All_probe_angles_images_foot2.append(probe_018_images_foot2)
All_probe_angles_infos_foot2.append(probe_018_infos_foot2)

All_probe_angles_images_foot2.append(probe_036_images_foot2)
All_probe_angles_infos_foot2.append(probe_036_infos_foot2)

All_probe_angles_images_foot2.append(probe_054_images_foot2)
All_probe_angles_infos_foot2.append(probe_054_infos_foot2)

All_probe_angles_images_foot2.append(probe_072_images_foot2)
All_probe_angles_infos_foot2.append(probe_072_infos_foot2)

All_probe_angles_images_foot2.append(probe_090_images_foot2)
All_probe_angles_infos_foot2.append(probe_090_infos_foot2)

All_probe_angles_images_foot2.append(probe_108_images_foot2)
All_probe_angles_infos_foot2.append(probe_108_infos_foot2)

All_probe_angles_images_foot2.append(probe_126_images_foot2)
All_probe_angles_infos_foot2.append(probe_126_infos_foot2)

All_probe_angles_images_foot2.append(probe_144_images_foot2)
All_probe_angles_infos_foot2.append(probe_144_infos_foot2)

All_probe_angles_images_foot2.append(probe_162_images_foot2)
All_probe_angles_infos_foot2.append(probe_162_infos_foot2)

All_probe_angles_images_foot2.append(probe_180_images_foot2)
All_probe_angles_infos_foot2.append(probe_180_infos_foot2)







# %%
from torch.nn import functional as F
from torch import Tensor
class LayerNorm2d(torch.nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

# %%
######MODEL
class Net(torch.nn.Module):
    def __init__(self,output_dim):
        super(Net, self).__init__()

        # Load pretrained convnext_base model
        self.pretrained_model = models.convnext_base(weights= models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Modify the classifier
        self.pretrained_model.classifier = torch.nn.Sequential(
            LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(1024, 4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, output_dim),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        features = self.pretrained_model.features(x)
        avg_pooled = self.pretrained_model.avgpool(features)
        return avg_pooled

# %%
import torch.nn.functional as F
class SimilarityCalculator:
    def __init__(self,query_vector,test_vector):
        self.query_vector=torch.tensor(query_vector,dtype=torch.float32).to(device)
        self.test_vector=torch.tensor(test_vector,dtype=torch.float32).to(device)
        self.query_features=self.query_vector.unsqueeze(0)
        self.test_features=self.test_vector.unsqueeze(0)
    
    def calculate_similarity(self):
        query_features_norm= F.normalize(self.query_features,p=2,dim=1)
        test_features_norm= F.normalize(self.test_features,p=2,dim=1)

        result=torch.matmul(query_features_norm,test_features_norm.t())
        return result.item()

# %%
class CustomDataset(Dataset):
    def __init__(self, list1, list2, list3, list4, list5):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.list4 = list4
        self.list5 = list5

    def __len__(self):
        return len(self.list1)

    def __getitem__(self, idx):
        list1 = self.list1[idx]
        list2 = self.list2[idx]
        list3 = self.list3[idx]
        list4 = self.list4[idx]
        list5 = self.list5[idx]
        return list1, list2, list3, list4, list5

# %%
def hesapla(img1,net):
  with torch.no_grad():
    x=img1.reshape((1,3,224,224)).to(device)
    feature= net(x)
    del x
    
  return np.squeeze(feature)

# %%
all_models_head = []
all_models_chest1 = []
all_models_chest2 = []
all_models_foot = []
all_models_foot2 = []

if torch.cuda.is_available():
    torch.cuda.empty_cache()

for i in range(5):  
    model_ens_head = Net(74)  
    # model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/kfold/models_kfold_PART5/ensemble_PART1_model__{k}.pth'))
    model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/kfold/models_kfold_PART1/ensemble_PART1_model__{k}.pth'))
    # model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/bootstrap/models_bootsrap_PART1/ensemble_PART1_model__{k}.pth'))
    # model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/bootstrap/models_bootsrap_PART1/ensemble_PART1_model__{k}.pth'))


    with torch.no_grad():
        model_ens_head.eval() 
    all_models_head.append(model_ens_head)

for i in range(5):  
    model_ens_chest1 = Net(74)  
    # model_ens_chest1.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/kfold/models_kfold_PART5/ensemble_PART2_model__{k}.pth'))
    model_ens_chest1.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/kfold/models_kfold_PART2/ensemble_PART2_model__{k}.pth'))
    # model_ens_chest1.load_state_dict(torch.load(fD:/WEIGHTS/CASIA_B/5PART/bootstrap/models_bootsrap_PART2/ensemble_PART2_model__{k}.pth'))
    # model_ens_chest1.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/bootstrap/models_bootsrap_PART2/ensemble_PART2_model__{k}.pth'))

    with torch.no_grad():
        model_ens_chest1.eval() 
    all_models_chest1.append(model_ens_chest1)

for i in range(5):  
    model_ens_chest2 = Net(74)  
    # model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/kfold/models_kfold_PART5/ensemble_PART3_model__{k}.pth'))
    model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/kfold/models_kfold_PART3/ensemble_PART3_model__{k}.pth'))
    # model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/bootstrap/models_bootsrap_PART3/ensemble_PART3_model__{k}.pth'))
    # model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/bootstrap/models_bootsrap_PART3/ensemble_PART3_model__{k}.pth'))

    with torch.no_grad():
        model_ens_chest2.eval() 
    all_models_chest2.append(model_ens_chest2)

for i in range(5):  
    model_ens_foot = Net(74)  
    # model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/kfold/models_kfold_PART5/ensemble_PART4_model__{k}.pth'))
    model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/kfold/models_kfold_PART4/ensemble_PART4_model__{k}.pth'))
    # model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/bootstrap/models_bootsrap_PART4/ensemble_PART4_model__{k}.pth'))
    # model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/bootstrap/models_bootsrap_PART4/ensemble_PART4_model__{k}.pth'))


    with torch.no_grad():
        model_ens_foot.eval() 
    all_models_foot.append(model_ens_foot)

for i in range(5):  
    model_ens_foot2 = Net(74)  
    # model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/kfold/models_kfold_PART5/ensemble_PART5_model__{k}.pth'))
    model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/kfold/models_kfold_PART5/ensemble_PART5_model__{k}.pth'))
    # model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART/bootstrap/models_bootsrap_PART5/ensemble_PART5_model__{k}.pth'))
    # model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/CASIA_B/5PART_WITHOUT_VARIATION/bootstrap/models_bootsrap_PART5/ensemble_PART5_model__{k}.pth'))


    with torch.no_grad():
        model_ens_foot2.eval() 
    all_models_foot2.append(model_ens_foot2)

# %%
############################## 5 models output for MetaModel
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import mode
from torch.utils.data import DataLoader

class MetaModel(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, num_models=5):
        super(MetaModel, self).__init__()
        self.fc = nn.Linear(num_models * input_dim, output_dim)
    
    def forward(self, features):
        combined_features = features.view(features.size(0), -1) 
        output = self.fc(combined_features) 
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta_model = MetaModel(input_dim=1024, num_models=len(all_models_head)).to(device)

# 
# %%
result_table=np.zeros((11, 11))

all_dataset_probes = CustomDataset(All_probe_angles_images_head,All_probe_angles_images_chest1,All_probe_angles_images_chest2, All_probe_angles_images_foot,All_probe_angles_images_foot2)
all_dataloader_probes = DataLoader(all_dataset_probes, batch_size=1, shuffle=False)

for i, (probe_head, probe_chest1, probe_chest2, probe_foot, probe_foot2) in enumerate(all_dataloader_probes): 
    dataset_probes = CustomDataset(probe_head, probe_chest1,probe_chest2,probe_foot, probe_foot2)
    dataloader_probes = DataLoader(dataset_probes, batch_size=1, shuffle=False)

    query_features_list_head = [[] for _ in range(len(all_models_head))]
    query_features_list_chest1 = [[] for _ in range(len(all_models_chest1))]
    query_features_list_chest2 = [[] for _ in range(len(all_models_chest2))]
    query_features_list_foot = [[] for _ in range(len(all_models_foot))]
    query_features_list_foot2 = [[] for _ in range(len(all_models_foot2))]



    for h in range(len(all_models_head)): 
        query_features_head = [torch.tensor(hesapla(img1, all_models_head[h].to(device))).to(device) if img1.numel() != 0 else torch.zeros(1024).to(device) for (img1, img2, img3, img4, img5) in dataloader_probes]
        query_features_list_head[h] = torch.stack(query_features_head)  
    
    for h in range(len(all_models_chest1)): 
        query_features_chest1 = [torch.tensor(hesapla(img2, all_models_chest1[h].to(device))).to(device) if img2.numel() != 0 else torch.zeros(1024).to(device) for (img1, img2, img3, img4, img5) in dataloader_probes]
        query_features_list_chest1[h] = torch.stack(query_features_chest1)  

    for h in range(len(all_models_chest2)): 
        query_features_chest2 = [torch.tensor(hesapla(img3, all_models_chest2[h].to(device))).to(device) if img3.numel() != 0 else torch.zeros(1024).to(device) for (img1, img2, img3, img4, img5) in dataloader_probes]
        query_features_list_chest2[h] = torch.stack(query_features_chest2) 
    
    for h in range(len(all_models_foot)): 
        query_features_foot = [torch.tensor(hesapla(img4, all_models_foot[h].to(device))).to(device) if img4.numel() != 0 else torch.zeros(1024).to(device) for (img1, img2, img3, img4, img5) in dataloader_probes]
        query_features_list_foot[h] = torch.stack(query_features_foot) 
    
    for h in range(len(all_models_foot2)): 
        query_features_foot2 = [torch.tensor(hesapla(img5, all_models_foot2[h].to(device))).to(device) if img5.numel() != 0 else torch.zeros(1024).to(device) for (img1, img2, img3, img4, img5) in dataloader_probes]
        query_features_list_foot2[h] = torch.stack(query_features_foot2) 


###############################  galleryler


    all_dataset_galleries = CustomDataset(All_gallery_angles_images_head, All_gallery_angles_images_chest1,All_gallery_angles_images_chest2,All_gallery_angles_images_foot, All_gallery_angles_images_foot2)
    all_dataloader_galleries = DataLoader(all_dataset_galleries, batch_size=1, shuffle=False)

    for j, (gallery_head, gallery_chest1, gallery_chest2, gallery_foot,gallery_foot2) in enumerate(all_dataloader_galleries):    
        dataset_galleries = CustomDataset(gallery_head, gallery_chest1, gallery_chest2, gallery_foot, gallery_foot2)
        dataloader_galleries = DataLoader(dataset_galleries, batch_size=1, shuffle=False) 


        gallery_features_list_head = [[] for _ in range(len(all_models_head))]
        gallery_features_list_chest1 = [[] for _ in range(len(all_models_chest1))]
        gallery_features_list_chest2 = [[] for _ in range(len(all_models_chest2))]
        gallery_features_list_foot = [[] for _ in range(len(all_models_foot))]
        gallery_features_list_foot2 = [[] for _ in range(len(all_models_foot2))]


        for h in range(len(all_models_head)): 
            gallery_features_head = [torch.tensor(hesapla(img1, all_models_head[h].to(device))).to(device) for (img1, img2, img3, img4, img5) in dataloader_galleries]
            gallery_features_list_head[h] = torch.stack(gallery_features_head)  
    
        for h in range(len(all_models_chest1)): 
            gallery_features_chest1 = [torch.tensor(hesapla(img2, all_models_chest1[h].to(device))).to(device) for (img1, img2, img3, img4, img5) in dataloader_galleries]
            gallery_features_list_chest1[h] = torch.stack(gallery_features_chest1)  

        for h in range(len(all_models_chest2)): 
            gallery_features_chest2 = [torch.tensor(hesapla(img3, all_models_chest2[h].to(device))).to(device) for (img1, img2, img3, img4, img5) in dataloader_galleries]
            gallery_features_list_chest2[h] = torch.stack(gallery_features_chest2) 
        
        for h in range(len(all_models_foot)): 
            gallery_features_foot = [torch.tensor(hesapla(img4, all_models_foot[h].to(device))).to(device) for (img1, img2, img3, img4, img5) in dataloader_galleries]
            gallery_features_list_foot[h] = torch.stack(gallery_features_foot) 
        
        for h in range(len(all_models_foot2)): 
            gallery_features_foot2 = [torch.tensor(hesapla(img5, all_models_foot2[h].to(device))).to(device) for (img1, img2, img3, img4, img5) in dataloader_galleries]
            gallery_features_list_foot2[h] = torch.stack(gallery_features_foot2) 



        if all(tensor.numel() != 0 for tensor in query_features_list_head):
            valid_query_features_head_list = query_features_list_head
            valid_gallery_features_head_list = gallery_features_list_head
        else:
            pass

        if all(tensor.numel() != 0 for tensor in query_features_list_chest1):
            valid_query_features_chest1_list = query_features_list_chest1
            valid_gallery_features_chest1_list = gallery_features_list_chest1
        else:
            pass

        if all(tensor.numel() != 0 for tensor in query_features_list_chest2):
            valid_query_features_chest2_list = query_features_list_chest2
            valid_gallery_features_chest2_list = gallery_features_list_chest2
        else:
            pass

        if all(tensor.numel() != 0 for tensor in query_features_list_foot):
            valid_query_features_foot_list = query_features_list_foot
            valid_gallery_features_foot_list = gallery_features_list_foot
        else:
            pass

        if all(tensor.numel() != 0 for tensor in query_features_list_foot2):
            valid_query_features_foot2_list = query_features_list_foot2
            valid_gallery_features_foot2_list = gallery_features_list_foot2
        else:
            pass


        if all(tensor.numel() != 0 for tensor in valid_query_features_head_list):
            combined_query_features_head = torch.cat(valid_query_features_head_list, dim=1).to(device)
            combined_gallery_features_head = torch.cat(valid_gallery_features_head_list, dim=1).to(device)

        if all(tensor.numel() != 0 for tensor in valid_query_features_chest1_list):
            combined_query_features_chest1 = torch.cat(valid_query_features_chest1_list, dim=1).to(device)
            combined_gallery_features_chest1 = torch.cat(valid_gallery_features_chest1_list, dim=1).to(device)

        if all(tensor.numel() != 0 for tensor in valid_query_features_chest2_list):
            combined_query_features_chest2 = torch.cat(valid_query_features_chest2_list, dim=1).to(device)
            combined_gallery_features_chest2 = torch.cat(valid_gallery_features_chest2_list, dim=1).to(device)

        if all(tensor.numel() != 0 for tensor in valid_query_features_foot_list):
            combined_query_features_foot = torch.cat(valid_query_features_foot_list, dim=1).to(device)
            combined_gallery_features_foot = torch.cat(valid_gallery_features_foot_list, dim=1).to(device)

        if all(tensor.numel() != 0 for tensor in valid_query_features_foot2_list):
            combined_query_features_foot2 = torch.cat(valid_query_features_foot2_list, dim=1).to(device)
            combined_gallery_features_foot2 = torch.cat(valid_gallery_features_foot2_list, dim=1).to(device)


        if all(tensor.numel() != 0 for tensor in combined_query_features_head):
            final_query_feature_head = meta_model(combined_query_features_head)
            final_gallery_feature_head = meta_model(combined_gallery_features_head)
        else:
            final_query_feature_head = torch.empty(0)
            final_gallery_feature_head = torch.empty(0)

        if all(tensor.numel() != 0 for tensor in combined_query_features_chest1):
            final_query_feature_chest1 = meta_model(combined_query_features_chest1)
            final_gallery_feature_chest1 = meta_model(combined_gallery_features_chest1)
        else:
            final_query_feature_chest1 = torch.empty(0)
            final_gallery_feature_chest1 = torch.empty(0)

        if all(tensor.numel() != 0 for tensor in combined_query_features_chest2):
            final_query_feature_chest2 = meta_model(combined_query_features_chest2)
            final_gallery_feature_chest2 = meta_model(combined_gallery_features_chest2)
        else:
            final_query_feature_chest2 = torch.empty(0)
            final_gallery_feature_chest2 = torch.empty(0)

        if all(tensor.numel() != 0 for tensor in combined_query_features_foot):
            final_query_feature_foot = meta_model(combined_query_features_foot)
            final_gallery_feature_foot = meta_model(combined_gallery_features_foot)
        else:
            final_query_feature_foot = torch.empty(0)
            final_gallery_feature_foot = torch.empty(0)

        if all(tensor.numel() != 0 for tensor in combined_query_features_foot2):
            final_query_feature_foot2 = meta_model(combined_query_features_foot2)
            final_gallery_feature_foot2 = meta_model(combined_gallery_features_foot2)
        else:
            final_query_feature_foot2 = torch.empty(0)
            final_gallery_feature_foot2 = torch.empty(0)



        result = np.zeros((len(final_query_feature_head), len(final_gallery_feature_head)))

        for k, query_features in enumerate(zip(final_query_feature_head, final_query_feature_chest1, final_query_feature_chest2, final_query_feature_foot, final_query_feature_foot2)):
            non_empty_query_features = [feature for feature in query_features if feature.numel() > 0]

            for l, gallery_features in enumerate(zip(final_gallery_feature_head, final_gallery_feature_chest1, final_gallery_feature_chest2, final_gallery_feature_foot, final_gallery_feature_foot2)):
                non_empty_gallery_features = [feature for feature in gallery_features if feature.numel() > 0]

                if len(non_empty_query_features) == len(non_empty_gallery_features):
                    fused_output_query = torch.cat(non_empty_query_features, dim=0)
                    fused_output_gallery = torch.cat(non_empty_gallery_features, dim=0)
                    cos_sim = SimilarityCalculator(fused_output_gallery, fused_output_query)
                    result[k][l] = cos_sim.calculate_similarity()

        
        Probe_num=len(All_probe_angles_images_foot[i])
        tmp_probe_infos=[0]*len(All_probe_angles_images_foot[i])

        indexes=[]
        for m in range(result.shape[0]):
            index=np.argmax(result[m])
            tmp_probe_infos[m]= All_gallery_angles_infos_foot[j][index]
            indexes.append(index)

        true_count=0
        for n in range(len(tmp_probe_infos)):
            if tmp_probe_infos[n][0]==All_probe_angles_infos_foot[i][n][0]:
                true_count=true_count+1

        success=100*true_count/Probe_num
        result_table[i][j]=success
        print(result_table[i][j])
        success=0
    print("------")

# %%
print(result_table)


# %%
# array saving
import pickle
with open('results/bootsrap/NM_ensemble.pkl', 'wb') as f:
    pickle.dump(result_table, f)
    f.close()


