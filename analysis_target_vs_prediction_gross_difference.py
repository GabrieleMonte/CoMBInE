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
from importlib.machinery import SourceFileLoader
mask = SourceFileLoader('Mask_generator.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator.py')).load_module()
unet_100 = SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()
start_time = time.time()
# Settings
BATCH_SIZE = 3


# Instantiate mask generator (random shape masks)
mask_generator = mask.MaskGenerator(128, 128, 3, rand_seed=42)


#create directories for analysis plots and data files
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/', exist_ok=True)
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/', exist_ok=True)

#create list with all the models corresponding to each epoch fro step 1 and step2
file_tot=list()
file1=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'), 'weights.*.h5')
file2=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/'), 'weights.*.h5')
file_tot.append(file1)
file_tot.append(file2)
file_len=len(file_tot[0])+len(file_tot[1]) #compute lenght of the file= # of epochs of training completed

#create list of all images from validation set
img_files=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_maps_cropped_center_sky/Validation/CMB_maps_cropped'), 'CMB_map_cropped_*.jpg')

#instatiate model with pythorc VGG 16 image-net weights
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')


#Definition of useful arrays:
img=np.zeros((50,128,128,3)) #array that will contain the raw images
pred_img=np.zeros((file_len,50,128,128,3)) #array that will contain the predicted images
md=np.zeros((file_len,50)) #mean difference
pd=np.zeros((file_len,50)) #percentage mean difference
pd_std=np.zeros((file_len,50)) #percentage standard deviation from mean
md_std=np.zeros((file_len,50)) #standard deviation from mean 
x=np.arange(file_len) #array that identifies number of epoch

#Mask generation + application of the mask:
mask=mask_generator.sample()
for j in range(50):
  # Load 50 random images from validation set
  img[j]= np.array(Image.open('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_maps_cropped_center_sky/Validation/CMB_maps_cropped/'+str(img_files[np.random.randint(0,1500)])).resize((128, 128))) / 255

#fill array of masked images
masked_img = [ i for i in copy.deepcopy(img)]
for j in range(50):
  masked_img[j][mask==0]=1


#Calculation after each epoch of the values defined earlier related to the difference betweeen target and predicted image:
def variable_calc(i):
  for j in range(50):
    md[i][j]=np.mean(np.sqrt(((pred_img[i][j])-(img[j]))**2))
    pd[i][j]=md[i][j]/np.mean(img[j])*100
    md_std[i][j]=np.std(np.sqrt(((pred_img[i][j])-(img[j]))**2))


#Plot such values defined earlier for all images at each epoch:
def plotterino(md, pd, md_std,pd_std,x):
  y1=md.transpose()
  y2=pd.transpose()
  y3=md_std.transpose()
  y4=pd_std.transpose()
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300)
  for i in trange(50):
    plt.scatter(x, y1[i],c='red')
    plt.scatter(x, y3[i],c='green')
  plt.title('Difference between Predicted and Target img: \n Mean and Std Dev (Absolute value); \n train_set=10000 imgs')
  plt.xlabel('# of Epoch')
  plt.ylabel('Absolute Value')
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_abs-value_plot1_phase1_trainset_10000.jpg')
  plt.close()
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300) 
  for i in trange(50):
    plt.scatter(x, y2[i], c='red')
  plt.xlabel('# of Epoch')
  plt.ylabel('Percentage')
  plt.title('Difference between predicted and target img: \n Mean (Percentage); \n train_set=10000 imgs')
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_percent_plot1_phase1_trainset_100000.jpg')
  plt.close()


#Execute prediction for each model corresponding to a different epoch:  
for i in trange(len(file_tot[0])):
  model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'+ str(file1[i]))
  # Run predictions for this batch of images
  pred_img[i] = [model.predict([ np.expand_dims(j, 0), np.expand_dims(mask, 0)])[0] for j in masked_img]
  variable_calc(i)

for i in trange(len(file_tot[1])):
  model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/'+ str(file2[i]))
  # Run predictions for this batch of images
  pred_img[i+len(file_tot[0])] = [model.predict([ np.expand_dims(j, 0), np.expand_dims(mask, 0)])[0] for j in masked_img]
  variable_calc(i+len(file_tot[0]))
  #mean_difff=mean_diff[i]


plotterino(md, pd, md_std, pd_std, x)

#Evaluate mean and standard deviation of the values defined earlier at each epoch to create Error Bar Plots
y11=[np.mean(i) for i in md]
y12=[np.std(i) for i in md]
y21=[np.mean(i) for i in pd]
y22=[np.std(i) for i in pd]
y31=[np.mean(i) for i in md_std]
y32=[np.std(i) for i in md_std]

#Error bar plot for diffenrece between target and prediction of the mean and std dev in absolute value and of the mean in percentage
plt.figure(figsize=(12.8, 12.8), dpi=300)
plt.subplot(211)
plt.title('Difference between Predicted and Target img: \n train_set=10000 imgs \n Mean and Std Dev (Absolute value)')
plt.errorbar(x, y11, yerr=y12, ecolor='black', fmt='bo', label='mean diff.', linestyle='None', alpha=0.5)
plt.errorbar(x, y31, yerr=y32, ecolor='grey',fmt='ko', label='std. dev. from mean diff.', linestyle='None', alpha=0.5)
plt.xlabel('# of Epoch')
plt.ylabel('Absolute Value')
plt.legend()
plt.subplot(212)
plt.title('\n Mean (Percentage)')
plt.errorbar(x, y21, yerr=y22, ecolor='black', fmt='bo', label='mean diff. in %', linestyle='None', alpha=0.5)
plt.xlabel('# of Epoch')
plt.ylabel('Percentage Value')
plt.legend()
plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_percent_plot2_phase1_trainset_10000.jpg')
plt.close()


#Save all values in text files for further analysis:
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/md_hist.txt', md)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/pd_hist.txt', pd)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/md_std_hist.txt', md_std)
np.savetxt('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/data_files/pd_std_hist.txt', pd_std)


print("--- %s seconds ---" % (time.time() - start_time))


