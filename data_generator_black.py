# A python script to generate the data
# The script generates six classes of data with the corresponding labels.
# The sixth class is the noise class. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import hamming
import math
import random
import pandas as pd
import os
import argparse
import time
import cv2
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
    #print('XDIS YDIS', x_dis, y_dis)
    distorted = np.full(atom_positions.shape, fill_value=[x_dis, y_dis], dtype=float)
    #print('ATOM+DIS \n', atom_positions+dis)
    return distorted + atom_positions


def position_generator(lattice_structure, nx = 20, ny = 20):

    """
    A function to generate the positions of atoms in the lattice. The atom positions
    generated by this function do not include the perturbations.

    Parameters
    -----------
    lattice_structure: str
        A string specifying the lattice structure you want. 

    Returns
    -----------
    atom_pos: np.array
        The atom positions for one image of the specified class
    """

    a1 = None
    a2 = None
    phi = None

    if lattice_structure=='hexagonal':
        a1 = a2 = random.uniform(0.8, 2)
        phi = 2*(math.pi/3)
    
    elif lattice_structure=='square':
        a1 = a2 = random.uniform(0.8, 2)
        phi = math.pi/2
    
    elif lattice_structure=='rectangular':
        a2 = random.uniform(0.8, 2)
        a1 = a2*random.uniform(1.3, 1.6)
        phi = math.pi/2
    
    elif lattice_structure=='oblique':
        a1 = random.uniform(0.8, 2)
        a2 = random.uniform(0.8, 2)
        phi = random.uniform(0, math.pi)
        while(math.isclose(phi, math.pi/2, abs_tol=1e-3)):
            phi = random.uniform(0, math.pi)
    
    elif lattice_structure=='noise':
        a1 = random.uniform(0.8, 2)
        a2 = random.uniform(0.8, 2)
        phi = random.uniform(0, math.pi)
        while(math.isclose(phi, math.pi/2, abs_tol=1e-3)):
            phi = random.uniform(0, math.pi)
    
    else:
        a1 = a2 = random.uniform(0.8, 2)
        phi = random.uniform(0, math.pi)
        while(math.isclose(phi, math.pi/2, abs_tol=1e-1)):
            phi = random.uniform(0, math.pi)
    


    nx = ny = 20
    nx, ny = np.meshgrid(np.arange(nx), np.arange(ny))
    atom_pos = []
    for nxx, nyy in zip(nx.ravel(), ny.ravel()):
        x_ind = nxx*a1 +nyy*a2*np.cos(phi)
        y_ind = nyy*a2*np.sin(phi)
        atom_pos.append((x_ind, y_ind))
    atom_pos = np.array(atom_pos)
    return atom_pos, a1, a2

def save_image(directory, structure, atom_pos, index):

    """
    A function to save the lattice images produced into a folder.

    Parameters
    -----------
    directory: str
        Where to store the images
    atom_pos: array
        The positions of all the atoms in the lattice.  
    index: int
        The index of the lattice (an integer to uniquely identify the lattice image in the folder).
    structure: str
        A string specifying the lattice structure.
    """

    image_atoms = np.zeros((1024, 1024))
    max_x = np.max(atom_pos[:,0])
    max_y = np.max(atom_pos[:,1])

    for ind in range(atom_pos.shape[0]):
        x1, y1 = atom_pos[ind, 0], atom_pos[ind, 1]
        x_img = int(x1/max_x * (image_atoms.shape[0]-1))
        y_img = int(y1/max_y * (image_atoms.shape[1]-1))

        try:
            if x_img>0 and y_img>0:
                image_atoms[x_img, y_img] = 1E6
        except:
            pass
                
    n = image_atoms.shape[0]
    h = 256

    h = hamming(n)
    ham2d = np.sqrt(np.outer(h,h))

    img_convolved = gaussian_filter(image_atoms, sigma = 6, order = 0)
    img_windowed = np.copy(img_convolved)
    img_windowed *= ham2d
    fft_win_size = 64

    img_fft = np.fft.fftshift(np.fft.fft2(img_windowed))
    final_image = np.sqrt(np.abs(img_fft[image_atoms.shape[0]//2 - fft_win_size:image_atoms.shape[0]//2+fft_win_size,
                                 image_atoms.shape[0]//2 - fft_win_size:image_atoms.shape[0]//2+fft_win_size]))
    if os.path.exists(directory+structure) == False:
        os.mkdir(directory+structure)
    plt.imsave(directory+structure+'/'+str(index)+'.png', final_image,format='png')
    new_img = cv2.imread(directory+structure+'/'+str(index)+'.png',1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(directory+structure+'/'+str(index)+'.png',gray)

def generate_train_data(data_size):
    directory = 'data/train/'
    structures = ['hexagonal','square','rectangular','centred', 'oblique', 'noise']
    ground_truth_id = []
    ground_truth_label = []
    total_counter = 0
    '''
    positions, a1, a2 = position_generator('hexagonal')
    distorted = distortions(positions, a1, a2, distortion_factor=0.03)
    print(distorted.shape)
    '''
    for structure in structures:
        if structure == 'noise':
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = add_noise(positions)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)
        else:
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = distortions(positions, a1, a2)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)

    data = pd.DataFrame.from_dict({'id': ground_truth_id, 'label':ground_truth_label})
    data.to_csv(directory+'ground_truth.csv',index = False)
    print('Generated '+str(total_counter)+' train images')
    

def generate_val_data(data_size):
    directory = 'data/val/'
    structures = ['hexagonal','square','rectangular','centred', 'oblique', 'noise']
    ground_truth_id = []
    ground_truth_label = []
    total_counter = 0
    '''
    positions, a1, a2 = position_generator('hexagonal')
    distorted = distortions(positions, a1, a2, distortion_factor=0.03)
    print(distorted.shape)
    '''
    for structure in structures:
        if structure == 'noise':
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = add_noise(positions)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)
        else:
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = distortions(positions, a1, a2)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)

    data = pd.DataFrame.from_dict({'id': ground_truth_id, 'label':ground_truth_label})
    data.to_csv(directory+'ground_truth.csv',index = False)
    print('Generated '+str(total_counter)+' validation data')

def generate_test_data(data_size):
    directory = 'data/test/'
    structures = ['hexagonal','square','rectangular','centred', 'oblique', 'noise']
    ground_truth_id = []
    ground_truth_label = []
    total_counter = 0
    '''
    positions, a1, a2 = position_generator('hexagonal')
    distorted = distortions(positions, a1, a2, distortion_factor=0.03)
    print(distorted.shape)
    '''
    for structure in structures:
        if structure == 'noise':
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = add_noise(positions)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)
        else:
            for i in range(0, int(data_size//6)):
                total_counter+=1
                positions, a1, a2 = position_generator(structure)
                distorted = distortions(positions, a1, a2)
                save_image(directory = directory, structure = structure, atom_pos = distorted, index = total_counter)
                ground_truth_id.append(total_counter)
                ground_truth_label.append(structure)

    data = pd.DataFrame.from_dict({'id': ground_truth_id, 'label':ground_truth_label})
    data.to_csv(directory+'ground_truth.csv',index = False)
    print('Generated '+str(total_counter)+' test images.')

def generate_data(data_size):
    
    generate_train_data(data_size*0.7)
    generate_val_data(data_size*0.3)
    generate_test_data(data_size*0)

if __name__ == '__main__':
    #print(parameters)
    start = time.time()
    generate_data(15000)
    end = time.time()
    print("Time taken = ", str(end-start))

