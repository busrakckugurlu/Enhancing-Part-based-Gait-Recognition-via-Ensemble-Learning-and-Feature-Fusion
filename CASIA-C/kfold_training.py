# %%
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm
from torchinfo import summary
from torchmetrics import Accuracy

from sklearn.model_selection import StratifiedKFold

# %%
def load_training_data(src_dir):
    id = ["{0:03}".format(i) for i in range(1, 63)]   #range(1, 101), range(1, 63) , range(1, 25)
    categories = ["fb00", "fb01", "fn00", "fn01","fn02", "fn03", "fq00", "fq01","fs00", "fs01"]
    im_images=[]
    lab_labels=[]
    for i in range(len(id)):
        for j in range(len(categories)):
            for l in os.listdir(os.path.join(src_dir, id[i], categories[j])):
                path=os.path.join(src_dir, id[i], categories[j], l)
                im_images.append(path)
                label = "{0:03}".format(int(l.split("_")[0])-1)
                # print(path)
                # print(label)
                lab_labels.append(label)
    
    lab_labels2 = np.zeros(len(lab_labels))
    for i in range(0, len(lab_labels)):
        lab_labels2[i] = int(lab_labels[i])
    lab_labels2 = np.array(lab_labels2)

    x_train = np.array(im_images)
    y_train = np.array([np.eye(62)[int(elm)] for elm in lab_labels2]).astype('float32')
    
    return x_train, y_train


#src_dir="D:/CASIA_C/5Part/1_PART"
#src_dir="D:/CASIA_C/5Part/2_PART"
#src_dir="D:/CASIA_C/5Part/3_PART"
#src_dir="D:/CASIA_C/5Part/4_PART"
src_dir="D:/CASIA_C/5Part/5_PART"


x_train, y_train = load_training_data(src_dir)

# %%
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# %%
class dataset(Dataset):
  def __init__(self,x,y,transform=None):
    self.x = x 
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
    self.transform = transform
 
  def __getitem__(self,idx):  
    if self.transform:
      if isinstance(idx, slice):
        images = [Image.open(img).convert('RGB') for img in self.x[idx]]
        timages = [self.transform(img) for img in images]
        return torch.stack(timages), self.y[idx]
      else:
        image = Image.open(self.x[idx]).convert('RGB') 
        return self.transform(image),self.y[idx]
    else:
      return self.x[idx],self.y[idx]

  def __len__(self):
    return self.length

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
###### MODEL
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
        out = self.pretrained_model(x)
        return out

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
output_dim = 62
batch_size = 4

# %%
#hyper parameters
learning_rate = 0.0001

loss_fn = torch.nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=74).to(device)

# %%
y_train_int = np.argmax(y_train, axis=1)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# %%
for k, (train_index, val_index) in enumerate(skf.split(x_train, y_train_int)):

    x_train_split, x_val_split = x_train[train_index], x_train[val_index]
    y_train_split, y_val_split = y_train[train_index], y_train[val_index]
    

    trainset = dataset(x_train_split,y_train_split,transform=preprocess)
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle = True)

    model=Net(output_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
    
    for param in model.parameters():
      param.requires_grad = True

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch in range(50): 
        with tqdm(trainloader, unit="batch", ascii=True) as tepoch:
            btrain_loss_list = []
            btrain_acc_list = []
            for i,(images,labels) in enumerate(tepoch):
            
                tepoch.set_description("Epoch {0:3d}".format(epoch))      
                outputs = model(images.to(device))
                train_loss = loss_fn(outputs,labels.to(device))
                predicted = torch.max(outputs.data,1)[1]
                ground_truth = torch.max(labels.to(device),1)[1]
                train_acc = accuracy(predicted,ground_truth) #(predicted == ground_truth).float().mean() 

                btrain_loss_list.append(train_loss.data.cpu().numpy())
                btrain_acc_list.append(train_acc.cpu().numpy())
                tepoch.set_postfix({'Train Loss':train_loss.data.cpu().numpy(), 'Train Accuracy':train_acc.cpu().numpy()})

                optimizer.zero_grad()  
                train_loss.backward()
                optimizer.step()      

                if(i == len(tepoch)-1):
                    mean_train_lost = np.array(btrain_loss_list).sum()/float(len(trainloader))
                    mean_train_acc = np.array(btrain_acc_list).sum()/float(len(trainloader))
                    train_loss_list.append(mean_train_lost)
                    train_acc_list.append(mean_train_acc)
                    tepoch.set_postfix({'Train Loss':mean_train_lost, 'Train Accuracy':mean_train_acc})

    torch.save(model.state_dict(), f'D:/WEIGHTS/Casia_C/5Part/Ensembles/models_kfold/Part5_models/ensemble_Part5_62id_model__{k}.pth')


# %%



