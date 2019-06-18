#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import hamming


"""This function creates the lattice points for the all the images in a single go
and stacks them in 3D"""
def lattice(parms,nxx=20, nyy=20):
  a1 = parms[0]
  a2 = parms[1]
  phi = parms[2]
  
  nx,ny = np.meshgrid(np.arange(nxx), np.arange(nyy))
  values = (nx.ravel(),ny.ravel())
  values = np.array(values).T
  nxx = values[:,0]
  nyy = values[:,1]
  j = a1.shape[0]
  atom_pos = np.zeros((j,400,2))
  for i in range(0,j):

      x_ind = nxx * a1[i] + nyy * a2[i] * np.cos(phi[i])
      y_ind = nyy * a2[i] * np.sin(phi[i])
      atom_pos[i,:,0] = x_ind
      atom_pos[i,:,1] = y_ind
  
  return atom_pos


"""This function distorts the positions of all the atoms in the lattice with
standard deviation 30% of the lattice parameter a1"""
def distortions(atom_pos, parms):
  j = atom_pos.shape[0]
  a1 = parms[0]
  a2 = parms[1]
  atom_pos_dis = np.zeros((j,400,2))
  for i in range(j):
    x_dis = np.random.normal(loc = 0.0, scale = 0.03*a1[i] ,size = [400,1])
    y_dis = np.random.normal(loc = 0.0, scale = 0.03*a2[i] ,size = [400,1])

    dis = np.concatenate((x_dis,y_dis),axis=1)
    atom_pos_dis[i] = atom_pos[i] + dis
  
  return atom_pos_dis


def atom_to_image(atom_pos_dis,img_dim = 1024):
  j = atom_pos_dis.shape[0]
  image_atoms = np.zeros([j,img_dim,img_dim])
    
  for i in range(0,j):
      max_x = np.max(atom_pos_dis[i,:,0])
      max_y = np.max(atom_pos_dis[i,:,1])

      min_x = np.min(atom_pos_dis[i,:,0])
      min_y = np.min(atom_pos_dis[i,:,1])



      x1,y1 = atom_pos_dis[i,:,0], atom_pos_dis[i,:,1]
      x_img = ((x1 - min_x)/(max_x - min_x) * (img_dim-1))       
      y_img = ((y1-min_y)/(max_y - min_y) * (img_dim-1))

      x_img = x_img.astype(int)
      y_img = y_img.astype(int)  
      image_atoms[i,x_img, y_img]=1E6           


  return image_atoms

def convolve_atomic_image(image_atoms, sigma = 6):
  j = image_atoms.shape[0]
  con_img = np.zeros([j,1024,1024])
  for i in range(0,j):
      con_img[i] = gaussian_filter(image_atoms[i],sigma,order = 0)
  return con_img

def crop(convolved_img):
    crop_img = convolved_img[:,350:700,350:700]
    
    
    return crop_img

def fft(crop_img):
    img_ffts=[]
    fft_win_size =64
    j = crop_img.shape[0]
    for j in range(0,j):
        
        n = crop_img.shape[1]
        h = hamming(n) 
        ham2d = np.sqrt(np.outer(h,h))
        img_windowed = np.copy(crop_img[j])
        img_windowed *= ham2d 
        img_fft = np.fft.fftshift(np.fft.fft2(img_windowed))
        img_fft = img_fft[crop_img.shape[1]//2 - fft_win_size:crop_img.shape[1]//2+fft_win_size,
                                     crop_img.shape[1]//2 - fft_win_size:crop_img.shape[1]//2+fft_win_size]
        img_ffts.append(img_fft)
        
    return np.array(np.sqrt(np.abs(img_ffts)))
  
  
  


# In[3]:


"""Debugged"""
import numpy as np
from sklearn.utils import shuffle

"""Generates 400 lattice parameters for each type of bravis lattice,
labels and shuffles them and output 100 data points when called"""

def generate():
#oblique   a1 != a2, phi!=90 Label = 0
    bond_len = np.random.uniform(low = [0.8,0.8], high = [2.0,2.2],size = [800,2])
    diff = np.abs(((bond_len[:,0]-bond_len[:,1])*100)/bond_len[:,1])

    bond_len = bond_len[diff > 20.0,:]
    bond_len = bond_len[0:400,:]
    phi = np.zeros((400,1))
    phi[0:300] = np.random.uniform(low = [0.0], high = [((55.0/180)*np.pi)],size = [300,1])
    phi[300:400] = np.random.uniform(low = [((65.0/180)*np.pi)], high = [((90.0/180)*np.pi)],size = [100,1])

    parms_obl = np.concatenate((bond_len,phi),axis = 1)
    
    #square  a1 = a2, phi = 90, Label = 1
    parms_sq = np.zeros([400,3])
    parms_sq[:,0] = np.random.uniform(low = 0.8, high = 2.0,size = 400)
    parms_sq[:,1] = parms_sq[:,0]
    parms_sq[:,2] = np.pi/2
    
    #Hexagonal a1 = a2, phi = 60 Label = 2
    parms_hex = np.zeros([400,3])
    parms_hex[:,0] = np.random.uniform(low = 0.8, high = 2.0,size = 400)
    parms_hex[:,1] = parms_hex[:,0]
    parms_hex[:,2] = (np.pi/3)
    
    #rectangular a1 != a2, phi = 90 Label =3
    parms_rec = np.random.uniform(low = [0.8,0.8,(np.pi/2)], high = [2.0,2.2,(np.pi/2)],size = [800,3])
    diff_rec = np.abs(((parms_rec[:,0]-parms_rec[:,1])*100)/parms_rec[:,1])
    
    parms_rec = parms_rec[diff_rec > 20.0,:]
    parms_rec = parms_rec[0:400,:]
    
    
    #centered  a1 = a2, phi != 90, Label = 4
    bond_len = np.zeros([400,2])
    bond_len[:,0] = np.random.uniform(low = 0.8, high = 2.0,size = 400)
    bond_len[:,1] = bond_len[:,0]
    phi = np.zeros((400,1))
    phi[0:300] = np.random.uniform(low = [0.0], high = [((55.0/180)*np.pi)],size = [300,1])
    phi[300:400] = np.random.uniform(low = [((65.0/180)*np.pi)], high = [((90.0/180)*np.pi)],size = [100,1])

    parms_cen = np.concatenate((bond_len,phi),axis = 1)
    
    parameters = np.concatenate((parms_obl,parms_sq,parms_hex,parms_rec, parms_cen), axis=0 )
    label = np.zeros([2000,1])
    
    label[401:801,0]=1
    label[801:1201,0]=2
    label[1201:1601,0]=3
    label[1601:2001,0]=4
    
    parameters,label = shuffle(parameters, label)
    parameters = parameters.T
    
    
    for i in range(0,10):
        parameters_list = parameters[:,i*100:(i+1)*100]
        labels_list = label[i*100:(i+1)*100,0]
        yield parameters_list, labels_list


# In[4]:


def images_final(parms):
  atom_pos = lattice(parms)
  atom_pos_dis = distortions(atom_pos, parms)
  image_atoms = atom_to_image(atom_pos_dis)
  con_img = convolve_atomic_image(image_atoms)
  crop_img = crop(con_img)
  img_final = fft(crop_img)

  return img_final


# In[5]:


import numpy as np
import torch, torchvision
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

a =  generate()
parameters_list, labels_list = next(a)
data = images_final(parameters_list)
data = data.astype(np.float32)

T = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
data_ten = T(data)
data_ten = data_ten.view(100,1,128,128)

labels_ten = torch.from_numpy(labels_list).long()


val_dataset = torch.utils.data.TensorDataset(data_ten,labels_ten)
val_dl = torch.utils.data.DataLoader(val_dataset,batch_size = 100)


class myCNN(nn.Module):
  def __init__(self):
    super(myCNN,self).__init__()
    self.cnn1 = nn.Conv2d(1,16,2)
    self.cnn2 = nn.Conv2d(16,32,2)
    self.cnn3 = nn.Conv2d(32,32,2)
    
    self.fc1 = nn.Linear(500000,128)
    self.fc2 = nn.Linear(128,50)
    self.fc3 = nn.Linear(50,10)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout2d()
    
  
  def forward(self,x):
    n = x.size(0)
    x = F.relu(self.cnn1(x))
    x = F.relu(self.cnn2(x))
    x = F.relu(self.cnn3(x))
    x = x.view(n,-1)
    
    
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = (self.fc3(x))
    
    return x
  
mycnn = myCNN().cuda()
cec = nn.CrossEntropyLoss()
optimizer = optim.SGD(mycnn.parameters(),lr = 0.001)


def validate(model,data):
  # To get validation accuracy = (correct/total)*100.
  total = 0
  correct = 0
  for k,(images,labels) in enumerate(data):
    images = Variable(images.cuda())
    x = model(images)
    value,pred = torch.max(x,1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)
  return correct*100./total

for j in range(18):
    parameters_list, labels_list = next(a)
    data = images_final(parameters_list)
    data = data.astype(np.float32)

    T = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_ten = T(data)
    data_ten = data_ten.view(100,1,128,128)

    labels_ten = torch.from_numpy(labels_list).long()


    train_dataset = torch.utils.data.TensorDataset(data_ten,labels_ten)
    train_dl = torch.utils.data.DataLoader(train_dataset,batch_size = 20)


    for e in range(35):
      for i,(images,labels) in enumerate(train_dl):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        pred = mycnn(images)
        loss = cec(pred,labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
          accuracy = float(validate(mycnn,val_dl))
          print('Epoch :',e+1,'Batch :',i+1,'Loss :',float(loss.data),'Accuracy :',accuracy,'%')


# In[ ]:


get_ipython().run_line_magic('reset', '')

