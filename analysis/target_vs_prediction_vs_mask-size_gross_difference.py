import os
import gc
import copy
from os import makedirs
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback


import matplotlib as mtpl
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import cv2
import time
from os.path import join
import fnmatch

#import libraries from GitHub
##NOTE: this command is only for Colaboratory notebooks, generally you can just import these libraries as any other common one by doing <import 'path of the library'>##
from importlib.machinery import SourceFileLoader
mask1 = SourceFileLoader('Mask_generator.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator.py')).load_module()
unet_100 = SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()
mask2 = SourceFileLoader('Mask_generator2.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator2.py')).load_module()
start_time = time.time()

# Settings
BATCH_SIZE = 3

#instatiate model with pytorch VGG 16 image-net weights
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')

#create list containing the model after each epoch (here we only upload the models from step 1 of the training)
file=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'), 'weights.*.h5')
# Load image (this analysis is conducted only on one image)
img= np.array(Image.open('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_maps_cropped_center_sky/Validation/CMB_maps_cropped/CMB_map_cropped_300_x235_y105.jpg').resize((128, 128))) / 255
 

# Instantiate mask generator (random shape masks)
mask_generator14 = mask1.MaskGenerator(128, 128, 3, rand_seed=42)

#Definition of useful arrays:

label=list()  #create list with the labels referring to the type of mask
mask_generator=list() #create list containing all the different types of mask
mask=np.zeros((14,128,128,3)) #array of the actual image of all the different types of mask
masked_img=np.zeros((14,128,128,3)) #array containing the image masked by all the different types of masks
pred_img=np.zeros((len(file),14,128,128,3)) #array for the predicted image for the varius masks after each epoch
md=np.zeros((len(file),14)) #mean difference
pd=np.zeros((len(file),14)) #percentage mean difference
pd_std=np.zeros((len(file),14)) #percentage standard deviation from mean
md_std=np.zeros((len(file),14)) #standard deviation from mean 
x=np.arange((len(file))) #array that identifies number of epoch

#Create mask_generator and label list + fill masked_img and mask array:     
for i in range(13):
  mask_generator.append(mask2.MaskGenerator2(128, 128, radius=(5*(i+1)),xcenter=64,ycenter=64, channels=3, rand_seed=42))
  mask[i]=mask_generator[i].sample()
  masked_img[i] =copy.deepcopy(img)
  label.append(str(5*(i+1)))
label.append('old mask')
mask_generator.append(mask_generator14)
mask[13]=mask_generator[13].sample()
masked_img[13] =copy.deepcopy(img)
for i in range(14):
  mask_circ=mask[i]
  masked_img[i][mask_circ==0]=1
  del mask_circ
  
#Calculation after each epoch of the values defined earlier related to the difference betweeen target and predicted image:def variable_calc(i):
  model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'+ str(file[i]))
  # Run predictions for this batch of images
  for j in range(14):
    pred_img[i][j]=model.predict([ np.expand_dims(masked_img[j], 0), np.expand_dims(mask[j], 0)])[0]
    md[i][j]=np.mean(np.sqrt(((pred_img[i][j])-(img))**2))
    pd[i][j]=md[i][j]/np.mean(img)*100
    md_std[i][j]=np.std(np.sqrt(((pred_img[i][j])-(img))**2))

#Plot such values defined earlier for all images at each epoch:
def plotterino(md, pd, md_std,pd_std,x):
  y1=md.transpose()
  y2=pd.transpose()
  y3=md_std.transpose()
  y4=pd_std.transpose()
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300)
  for i in trange(13):
    plt.scatter(x, y1[i],label=label[i])
  plt.plot(x, y1[13],label=label[13],color='black', marker='o', linestyle='dashed',linewidth=2)
  plt.title('Difference between Predicted and Target img for different Mask sizes: \n Mean (Absolute value); \n train_set=10000 imgs')
  plt.xlabel('# of Epoch')
  plt.ylabel('Absolute Mean Value')
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_mean_abs-value_plot3_phase1_trainset_10000.jpg')
  plt.close
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300)
  for i in trange(13):
    plt.scatter(x, y3[i],label=label[i])
  plt.plot(x, y3[13],label=label[13],color='black', marker='o', linestyle='dashed',linewidth=2)
  plt.title('Difference between Predicted and Target img for different Mask sizes: \n Standard Dev (Absolute value); \n train_set=10000 imgs')
  plt.xlabel('# of Epoch')
  plt.ylabel('Absolute Std Dev Value')
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_std-dev_abs-value_plot3_phase1_trainset_10000.jpg')
  plt.close()
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300) 
  for i in trange(13):
    plt.scatter(x, y2[i], label=label[i])
  plt.plot(x, y2[13],label=label[13],color='black', marker='o', linestyle='dashed',linewidth=2)
  plt.xlabel('# of Epoch')
  plt.ylabel('Percentage')
  plt.title('Difference between predicted and target img for different Mask sizes: \n Mean (Percentage); \n train_set=10000 imgs')
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_percent_plot3_phase1_trainset_100000.jpg')
  plt.close()

  
#Execute prediction for each model corresponding to a different epoch:  
for i in trange((len(file))):
  variable_calc(i)

  
plotterino(md, pd, md_std, pd_std, x)

#Save all values in text files for further analysis:
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/md_2.txt', md)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/pd_2.txt', pd)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/md_std_2.txt', md_std)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/pd_std_2.txt', pd_std)


print("--- %s seconds ---" % (time.time() - start_time))
