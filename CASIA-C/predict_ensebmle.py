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
    
    id = ["%03d" % i for i in range(100, 154)]
#gallery
    categories = ["fn00", "fn01","fn02", "fn03"]
    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    input_image = Image.open(path).convert("RGB")
                    input_tensor = preprocess(input_image)
                    #input_tensor=input_tensor.reshape((1,3,224,224))
                    gallery_images.append(input_tensor)
                    label = "{0:03}".format(int(l.split("_")[0])-1)
                    #print(label)
                    gallery_infos.append((label))
                    
#probe

    #categories = ["fb00", "fb01"]
    #categories = ["fq00", "fq01"]
    categories = ["fs00", "fs01"]

    for i in range(len(id)):
        for j in range(len(categories)):
                for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                    path=os.path.join(src_dir, id[i], categories[j], l)
                    input_image = Image.open(path).convert("RGB")
                    input_tensor = preprocess(input_image)
                    #input_tensor=input_tensor.reshape((1,3,224,224))
                    probe_images.append(input_tensor)
                    label = "{0:03}".format(int(l.split("_")[0])-1)
                    #print(label)
                    probe_infos.append((label))
                    
    return gallery_images, gallery_infos,probe_images,probe_infos


gallery_images_head, gallery_infos_head, probe_images_head, probe_infos_head = load_testing_data("D:/CASIA_C/5Part/1_PART")
gallery_images_chest1, gallery_infos_chest1, probe_images_chest1, probe_infos_chest1 = load_testing_data("D:/CASIA_C/5Part/2_PART")
gallery_images_chest2, gallery_infos_chest2, probe_images_chest2, probe_infos_chest2 = load_testing_data("D:/CASIA_C/5Part/3_PART")
gallery_images_foot, gallery_infos_foot, probe_images_foot, probe_infos_foot = load_testing_data("D:/CASIA_C/5Part/4_PART")
gallery_images_foot2, gallery_infos_foot2, probe_images_foot2, probe_infos_foot2 = load_testing_data("D:/CASIA_C/5Part/5_PART")


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
######burasııııı
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
        # Forward pass through the pretrained model
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
all_models_head = []
all_models_chest1 = []
all_models_chest2 = []
all_models_foot = []
all_models_foot2 = []

if torch.cuda.is_available():
    torch.cuda.empty_cache()

for i in range(5):  
    # model_ens_head = Net(24)  
    # model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part1_models/ensemble_Part1_24id_model__{i}.pth'))

    model_ens_head = Net(62)
    model_ens_head.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part1_models/ensemble_Part1_62id_model__{i}.pth'))

    with torch.no_grad():
        model_ens_head.eval() 
    all_models_head.append(model_ens_head)

for i in range(5):  
    # model_ens_chest1 = Net(24)  
    # model_ens_chest1.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part2_models/ensemble_Part2_24id_model__{i}.pth'))

    model_ens_chest1 = Net(62)  
    model_ens_chest1.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part2_models/ensemble_Part2_62id_model__{i}.pth'))

    with torch.no_grad():
        model_ens_chest1.eval() 
    all_models_chest1.append(model_ens_chest1)

for i in range(5):  
    # model_ens_chest2 = Net(24)  
    # model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part3_models/ensemble_Part3_24id_model__{i}.pth'))

    model_ens_chest2 = Net(62)  
    model_ens_chest2.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part3_models/ensemble_Part3_62id_model__{i}.pth'))

    with torch.no_grad():
        model_ens_chest2.eval() 
    all_models_chest2.append(model_ens_chest2)

for i in range(5):  
    # model_ens_foot = Net(24)  
    # model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part4_models/ensemble_Part4_24id_model__{i}.pth'))

    model_ens_foot = Net(62)  
    model_ens_foot.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part4_models/ensemble_Part4_62id_model__{i}.pth'))

    with torch.no_grad():
        model_ens_foot.eval() 
    all_models_foot.append(model_ens_foot)

for i in range(5):  

    # model_ens_foot2 = Net(24)  
    # model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part5_models/ensemble_Part5_24id_model__{i}.pth'))

    model_ens_foot2 = Net(62)  
    model_ens_foot2.load_state_dict(torch.load(f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part5_models/ensemble_Part5_62id_model__{i}.pth'))

    with torch.no_grad():
        model_ens_foot2.eval() 
    all_models_foot2.append(model_ens_foot2)

# %%
############################## METAMODEL
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

# %%
def hesapla(img1,net):
  with torch.no_grad():
    x=img1.reshape((1,3,224,224)).to(device)
    feature= net(x)
    del x
    
  return np.squeeze(feature)

# %%



query_features_list_head = [[] for _ in range(len(all_models_head))]
query_features_list_chest1 = [[] for _ in range(len(all_models_chest1))]
query_features_list_chest2 = [[] for _ in range(len(all_models_chest2))]
query_features_list_foot = [[] for _ in range(len(all_models_foot))]
query_features_list_foot2 = [[] for _ in range(len(all_models_foot2))]

for h in range(len(all_models_head)): 
    query_features_head = [torch.tensor(hesapla(img1, all_models_head[h].to(device))).to(device) for (img1) in probe_images_head]
    query_features_list_head[h] = torch.stack(query_features_head)  

for h in range(len(all_models_chest1)): 
    query_features_chest1 = [torch.tensor(hesapla(img2, all_models_chest1[h].to(device))).to(device) for (img2) in probe_images_chest1]
    query_features_list_chest1[h] = torch.stack(query_features_chest1)  

for h in range(len(all_models_chest2)): 
    query_features_chest2 = [torch.tensor(hesapla(img3, all_models_chest2[h].to(device))).to(device) for (img3) in probe_images_chest2]
    query_features_list_chest2[h] = torch.stack(query_features_chest2) 

for h in range(len(all_models_foot)): 
    query_features_foot = [torch.tensor(hesapla(img4, all_models_foot[h].to(device))).to(device) for (img4) in probe_images_foot]
    query_features_list_foot[h] = torch.stack(query_features_foot) 

for h in range(len(all_models_foot2)): 
    query_features_foot2 = [torch.tensor(hesapla(img5, all_models_foot2[h].to(device))).to(device) for (img5) in probe_images_foot2]
    query_features_list_foot2[h] = torch.stack(query_features_foot2) 


gallery_features_list_head = [[] for _ in range(len(all_models_head))]
gallery_features_list_chest1 = [[] for _ in range(len(all_models_chest1))]
gallery_features_list_chest2 = [[] for _ in range(len(all_models_chest2))]
gallery_features_list_foot = [[] for _ in range(len(all_models_foot))]
gallery_features_list_foot2 = [[] for _ in range(len(all_models_foot2))]

for h in range(len(all_models_head)): 
    gallery_features_head = [torch.tensor(hesapla(img1, all_models_head[h].to(device))).to(device) for (img1) in gallery_images_head]
    gallery_features_list_head[h] = torch.stack(gallery_features_head)  

for h in range(len(all_models_chest1)): 
    gallery_features_chest1 = [torch.tensor(hesapla(img2, all_models_chest1[h].to(device))).to(device) for (img2) in gallery_images_chest1]
    gallery_features_list_chest1[h] = torch.stack(gallery_features_chest1)  

for h in range(len(all_models_chest2)): 
    gallery_features_chest2 = [torch.tensor(hesapla(img3, all_models_chest2[h].to(device))).to(device) for (img3) in gallery_images_chest2]
    gallery_features_list_chest2[h] = torch.stack(gallery_features_chest2) 

for h in range(len(all_models_foot)): 
    gallery_features_foot = [torch.tensor(hesapla(img4, all_models_foot[h].to(device))).to(device) for (img4) in gallery_images_foot]
    gallery_features_list_foot[h] = torch.stack(gallery_features_foot) 

for h in range(len(all_models_foot2)): 
    gallery_features_foot2 = [torch.tensor(hesapla(img5, all_models_foot2[h].to(device))).to(device) for (img5) in gallery_images_foot2]
    gallery_features_list_foot2[h] = torch.stack(gallery_features_foot2) 



combined_query_features_head = torch.cat(query_features_list_head, dim=1).to(device)
combined_gallery_features_head = torch.cat(gallery_features_list_head, dim=1).to(device)

combined_query_features_chest1 = torch.cat(query_features_list_chest1, dim=1).to(device)
combined_gallery_features_chest1 = torch.cat(gallery_features_list_chest1, dim=1).to(device)

combined_query_features_chest2 = torch.cat(query_features_list_chest2, dim=1).to(device)
combined_gallery_features_chest2 = torch.cat(gallery_features_list_chest2, dim=1).to(device)

combined_query_features_foot = torch.cat(query_features_list_foot, dim=1).to(device)
combined_gallery_features_foot = torch.cat(gallery_features_list_foot, dim=1).to(device)

combined_query_features_foot2 = torch.cat(query_features_list_foot2, dim=1).to(device)
combined_gallery_features_foot2 = torch.cat(gallery_features_list_foot2, dim=1).to(device)


final_query_feature_head = meta_model(combined_query_features_head)
final_gallery_feature_head = meta_model(combined_gallery_features_head)

final_query_feature_chest1 = meta_model(combined_query_features_chest1)
final_gallery_feature_chest1 = meta_model(combined_gallery_features_chest1)

final_query_feature_chest2 = meta_model(combined_query_features_chest2)
final_gallery_feature_chest2 = meta_model(combined_gallery_features_chest2)

final_query_feature_foot = meta_model(combined_query_features_foot)
final_gallery_feature_foot = meta_model(combined_gallery_features_foot)

final_query_feature_foot2 = meta_model(combined_query_features_foot2)
final_gallery_feature_foot2 = meta_model(combined_gallery_features_foot2)

#################     
result = np.zeros((len(final_query_feature_head), len(final_gallery_feature_head)))

for k, (query_feature_h, query_feature_c1, query_feature_c2, query_feature_f,query_feature_f2) in enumerate(zip(final_query_feature_head,final_query_feature_chest1,final_query_feature_chest2,final_query_feature_foot,final_query_feature_foot2)):
    for l, (gallery_feature_h, gallery_feature_c1, gallery_feature_c2, gallery_feature_f,gallery_feature_f2) in enumerate(zip(final_gallery_feature_head,final_gallery_feature_chest1,final_gallery_feature_chest2,final_gallery_feature_foot, final_gallery_feature_foot2)):

        fused_output_query=torch.cat((query_feature_h, query_feature_c1, query_feature_c2, query_feature_f,query_feature_f2),dim=0)
        fused_output_gallery=torch.cat((gallery_feature_h, gallery_feature_c1, gallery_feature_c2, gallery_feature_f,gallery_feature_f2),dim=0)
        cos_sim=SimilarityCalculator(fused_output_gallery,fused_output_query)
        result[k][l] = cos_sim.calculate_similarity()


#################


Probe_num= len(probe_images_head)
tmp_probe_infos=[]
for k in range(len(probe_images_head)):
    tmp_probe_infos.append(0)
    
indexes=[]
for m in range(result.shape[0]):
    
    index=np.argmax(result[m])
    tmp_probe_infos[m]= gallery_infos_head[index] 
    indexes.append(index)
    
true_count=0
for n in range(len(tmp_probe_infos)):
    if tmp_probe_infos[n]==probe_infos_head[n]:   # gerçek etiket ile karşılaştırma
        true_count=true_count+1
        
        

success=100*true_count/Probe_num

print(success)

# %%



