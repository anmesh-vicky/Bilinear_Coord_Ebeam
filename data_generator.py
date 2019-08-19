#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import hamming


def add_noise(atom_positions):

    """
    This function generates data for the noise class.

    Parameters
    -----------
    atom_positions: array
        Array of atom_positions
    
    Returns
    -----------
    distorted: array
        Array of atoms positions after the noise has been added
    """
    noise = np.random.randn(atom_positions.shape[0], atom_positions.shape[1])
    return atom_positions+noise

def distortions(atom_positions, a1, a2, distortion_factor = 0.03):
    
    """
    This function adds distortions to the atom positions along the 'a1' lattice parameter.

    Parameters
    -----------
    atom_positions: array
        Array of atom_positions
    a1: float 
        The lattice parameter a1
    a2: float
        The lattice parameter a2
    distortion_factor: float
        The percentage by which the length between atoms has to be change
    
    Returns
    -----------
    distorted: array
        Array of atoms positions after the distortions
    """
    
    x_dis = np.random.normal(loc = 0.0, scale = distortion_factor*a1, size = None)
    y_dis = np.random.normal(loc = 0.0, scale = distortion_factor*a2, size = None)
#     print('XDIS YDIS', x_dis, y_dis)
    distorted = np.full(atom_positions.shape, fill_value=[x_dis, y_dis], dtype=float)
#     print("Before Distortion\n",atom_positions)
#     print('ATOM+DIS \n', atom_positions+distorted)
    return distorted+ atom_positions
def lattice(parms,nxx=25, nyy = 25):
    
    """Function that creates an oblique lattice
    Inputs: - parms:  3x1 vector with parameters [a1,a2,phi]
                      where a1 is the first lattice parameter, a2 is the second
                      and phi is the angle between the two vectors in radians
            - nxx: (optional), int, number of unit cells repeated in the x direction
            - nyy: (optional), int, number of unit cells repeated in the y direction
    
    Output: atom_positions: Numpy Matrix of size (nxx*nyy,2)"""
   
     
    a1 = parms[0]
    a2 = parms[1]
    phi = parms[2]

    #Create lattice

    nx,ny = np.meshgrid(np.arange(nxx), np.arange(nyy))

    atom_pos = []
    for nxx, nyy in zip(nx.ravel(),ny.ravel()):
        x_ind = nxx * a1 + nyy * a2 * np.cos(phi)
        y_ind = nyy * a2 * np.sin(phi)

        atom_pos.append((x_ind,y_ind))

    return np.array(atom_pos),a1 ,a2


def atom_to_img(atom_pos,img_dim = 1024, img_dim_1 = None,img_dim_2 = None):
    """Function that takes a list of atomic positions and converts them to an
    image
    Inputs: - atom_pos: output matrix of size (Nx2) with (x,y) coordinates of 
                        atoms in the lattice.
            - img_dim: (optional), int, size of image. Square images only.
            
    Output: atom_positions: Numpy Matrix of size (nxx*nyy,2)"""
    if img_dim_1:
        image_atoms = np.zeros((img_dim_1,img_dim_2))
    else:
        image_atoms = np.zeros((img_dim,img_dim))
    
    max_x = np.max(atom_pos[:,0])
    max_y = np.max(atom_pos[:,1])

    min_x = np.min(atom_pos[:,0])
    min_y = np.min(atom_pos[:,1])
    
    for ind in range(atom_pos.shape[0]):
        max_val = max(max_x, max_y)
        min_val = min(min_x, min_y)
        
        x1,y1 = atom_pos[ind,0], atom_pos[ind,1]
        if img_dim_1:
            x_img = int((x1 - min_val)/(max_val - min_val) * (img_dim_1-1)) #are we placing the images only at the integer coordinates?
            y_img = int((y1-min_val)/(max_val - min_val) * (img_dim_2-1))
        else:    
            x_img = int((x1 - min_val)/(max_val - min_val) * (img_dim-1)) #are we placing the images only at the integer coordinates?
            y_img = int((y1-min_val)/(max_val - min_val) * (img_dim-1))

        image_atoms[x_img, y_img]=1E6 #Place an intense point in the image at 
                                      #that atomic site
        
    return image_atoms

def convolve_atomic_img(image_atoms, sigma = 6):
# """Convolve input image with gaussian filter.
#   Input: - image_atoms: 2D Numpy Array
#          - sigma: int, optional, std. dev. of Gaussian
#   Output: - filtered_image: 2D Numpy array, same size as input image"""
  
    return gaussian_filter(image_atoms,sigma,order = 0)



####function that Genrates Images for differnet classes
def Params(Structure ,No_):
    if Structure == "Square":
        a1_list = np.random.uniform(low = 0.8, high =2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/2, high = np.pi/2, size = No_)
    elif Structure == "Rectangular":
        a1_list = np.random.uniform(low = 1, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/2, high = np.pi/2, size = No_)
        ######
    elif Structure == "Hexagonal":# fix
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/3, high = np.pi/3, size = No_)
    elif Structure == "Centred":
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = 1, high = np.pi/2-1e-3, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-3, high = 2.1 , size = No_//2))
    elif Structure == "Noise":      
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = np.random.uniform(low = 0.8, high = 2.1, size=No_)
        phi_list = np.random.uniform(low = 1, high = np.pi/2-1e-1, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-1, high = np.pi*.8-1e-1, size = No_//2))
    elif Structure == "Obilique":
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = .5, high = np.pi/2-1e-3, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-3, high = np.pi*.8-1e-1, size = No_//2))
#         print (phi_list)

    P = {"a1_list":a1_list,"a2_list":a2_list,"phi_list":phi_list}
    return P

####function that Genrates Images for differnet classes
def Params(Structure ,No_):
    if Structure == "Square":
        a1_list = np.random.uniform(low = 0.8, high =2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/2, high = np.pi/2, size = No_)
    elif Structure == "Rectangular":
        a1_list = np.random.uniform(low = 1, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/2, high = np.pi/2, size = No_)
        ######
    elif Structure == "Hexagonal":# fix
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = np.pi/3, high = np.pi/3, size = No_)
    elif Structure == "Centred":
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = 1, high = np.pi/2-1e-3, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-3, high = 2.1 , size = No_//2))
    elif Structure == "Noise":      
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = np.random.uniform(low = 0.8, high = 2.1, size=No_)
        phi_list = np.random.uniform(low = 1, high = np.pi/2-1e-1, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-3, high = np.pi*.8-1e-3, size = No_//2))
    elif Structure == "Obilique":
        a1_list = np.random.uniform(low = 0.8, high = 2.2,size = No_)
        a2_list = a1_list
        phi_list = np.random.uniform(low = .5, high = np.pi/2-1e-3, size = No_//2)
        phi_list = np.append(phi_list,np.random.uniform(low = np.pi/2+1e-3, high = np.pi*.8-1e-3, size = No_//2))
#         print (phi_list)

    P = {"a1_list":a1_list,"a2_list":a2_list,"phi_list":phi_list}
    return P



### data Generator #########
import os


def save_img(No_):
    directory = "./data_str/"
    Struct = ["Rectangular","Hexagonal","Centred","Square","Obilique","Noise"]
#     Struct = ["Centred"]
    for i in Struct:
        print("Current_Structure:"+i)
        img_ffts=[]
        fft_win_size = 128
        params = Params(i,No_)
        a1= params["a1_list"]
        a2 = params["a2_list"]
        phi = params["phi_list"]
       
      
#         print(a1_list-a2_list)
        index = 0
        ind = 0
#         print a1
        for ind in range(No_):
            parms_rhomb1 = [a1[ind], a2[ind], phi[ind]]
            latt,a1_,a2_ = lattice(parms_rhomb1)
            if i == "Noise":
                distorted = add_noise(latt)
                img_ = atom_to_img(distorted)
            elif i == "Square" or i == "Hexagonal" or i == "Centred":
                distorted = distortions(latt, a1_, a2_)
                img_ =np.random.random_integers(500,1024,1)[0]
                img_ = atom_to_img(distorted,img_dim =img_)             
            else:
                distorted = distortions(latt, a1_, a2_)
                img_1 =np.random.random_integers(500,1024,1)[0]
                img_2  = img_1 + np.random.random_integers(-200,200,1)[0]
                if  np.abs(img_1-img_2)<10 and (img_2>1024):
                    img_2  = img_1 + np.random.random_integers(-100,100,1)[0]
                img_ = atom_to_img(distorted,img_dim_1 = img_1,img_dim_2 =img_2)
            size = np.random.randint(low = 3, high = 7, size = 1)[0] 
            
            convolved_img = convolve_atomic_img(img_, sigma = size)
#             convolved_img = img_
            if (i == "Square" or i == "Hexagonal" or i == "Centred") and img_.shape[0]>700:
                convolved_img_cropped = convolved_img[img_.shape[0]//2-350:img_.shape[0]//2, img_.shape[0]//2-350:img_.shape[0]//2]
            elif img_.shape[0]<700 or img_.shape[1]<700  :
                min_size = min(img_.shape[0],img_.shape[1])
                convolved_img_cropped = convolved_img[:min_size,:min_size]
            else :
                convolved_img_cropped = convolved_img[350:700,350:700]
            
           #Calcualte the fft window
            n = convolved_img_cropped.shape[0]
            h = hamming(n) 
            ham2d = np.sqrt(np.outer(h,h)) 

           #Apply window
            img_windowed = np.copy(convolved_img_cropped)
            img_windowed *= ham2d 

           #Do the fft and append result
            img_fft = np.fft.fftshift(np.fft.fft2(img_windowed))
            img_fft = img_fft[convolved_img_cropped.shape[0]//2 - fft_win_size//2:convolved_img_cropped.shape[0]//2+fft_win_size//2,
                                     convolved_img_cropped.shape[0]//2 - fft_win_size//2:convolved_img_cropped.shape[0]//2+fft_win_size//2]
            img_ffts.append((img_fft,parms_rhomb1))
        for img in img_ffts:
            final_img = np.sqrt(np.abs(img[0]))
            if os.path.exists(directory+i) == False:
                    os.makedirs(directory+i)
            plt.imsave(directory+i+'/'+str(index)+'.png', final_img, format='png')
            index+=1


### generate 6000 images per class
save_img(6000)







