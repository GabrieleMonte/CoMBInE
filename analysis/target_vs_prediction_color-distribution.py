import os
import gc
import copy
from os import makedirs
import numpy as np
from PIL import Image
import itertools

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

import matplotlib as mtpl
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import cv2
import time
from os.path import join
import fnmatch
from scipy.stats import norm

#import libraries from GitHub
##NOTE: this command is only for Colaboratory notebooks, generally you can just import these libraries as any other common one by doing <import 'path of the library'>##
from importlib.machinery import SourceFileLoader
mask = SourceFileLoader('Mask_generator.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator.py')).load_module()
unet_100 = SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()
start_time = time.time()

# Instantiate mask generator
mask_generator = mask.MaskGenerator(128, 128, 3, rand_seed=42)

#Import random image from validation set
img= np.array(Image.open('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_maps_cropped_center_sky/Validation/CMB_maps_cropped/CMB_map_cropped_300_x235_y105.jpg').resize((128, 128))) / 255
#generate mask (random shape)
mask = mask_generator.sample()

# Image + mask
masked_img = copy.deepcopy(img)
masked_img[mask == 0] = 1

#create directory for analysis plots
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/', exist_ok=True)


#create list with all the models corresponding to each epoch fro step 1 and step2
file_tot=list()
file1=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'), 'weights.*.h5')
file2=fnmatch.filter(os.listdir('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/'), 'weights.*.h5')
file_tot.append(file1)
file_tot.append(file2)
file_len=len(file_tot[0])+len(file_tot[1]) #lenght of file list= # of epochs of training performed

#Instantiate model with pytorch VGG16 image-net weights
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')


#Definition of useful arrays:
pred_img=np.zeros((file_len,128,128,3)) #array that will contain the predicted images
r_img, g_img, b_img= list(), list(), list() #list that will contain the value of color red, blue and green in each pixel of the image
r_pred, g_pred, b_pred= np.zeros((file_len,16384)), np.zeros((file_len,16384)), np.zeros((file_len,16384)) #array that contains the predicted color value of red, green, blue in each pixel of predicted image per epoch
mu_r_pr, std_r_pr, mu_b_pr, std_b_pr, mu_g_pr, std_g_pr= np.zeros(file_len), np.zeros(file_len), np.zeros(file_len), np.zeros(file_len), np.zeros(file_len), np.zeros(file_len) #arrays of mean and standard deviation of gaussian distribution for R,G,B at each epoch
x=np.arange(file_len)

#fill array for color R,G,B in target image
for i,j in itertools.product(range(128),range(128)):
  r_img.append(img[i][j][0])
  g_img.append(img[i][j][1])
  b_img.append(img[i][j][2])



#evaluate array values for the predicted color R,G,B afte each epoch
def array_pred_calc(k): 
  r_pred1, g_pred1, b_pred1= list(), list(), list()
  for i,j in itertools.product(range(128),range(128)):
    r_pred1.append(pred_img[k][i][j][0])
    g_pred1.append(pred_img[k][i][j][1])
    b_pred1.append(pred_img[k][i][j][2])
  r_pred[k]=r_pred1
  b_pred[k]=b_pred1
  g_pred[k]=g_pred1
  del r_pred1, b_pred1, g_pred1

#compute mean and standard deviation of guassian distribution of R,G,B after each epoch  
def parameter_calc():
  for i in range(file_len):
    a1, a2= norm.fit(r_pred[i])
    a3, a4= norm.fit(b_pred[i])
    a5, a6= norm.fit(g_pred[i])
    mu_r_pr[i]=a1
    mu_b_pr[i]=a3
    mu_g_pr[i]=a5
    std_r_pr[i]=a2
    std_b_pr[i]=a4
    std_g_pr[i]=a6
    del a1, a2, a3, a4, a5, a6

#Execute prediction for each model corresponding to a different epoch: 
for i in trange(len(file_tot[0])):
  model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/'+ str(file1[i]))
  # Run predictions for this batch of images
  pred_img[i] = model.predict([ np.expand_dims(masked_img, 0), np.expand_dims(mask, 0)])[0]
  array_pred_calc(i)

for i in trange(len(file_tot[1])):
  model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/'+ str(file2[i]))
  # Run predictions for this batch of images
  pred_img[i+len(file_tot[0])] = model.predict([ np.expand_dims(masked_img, 0), np.expand_dims(mask, 0)])[0]
  array_pred_calc(i+len(file_tot[0]))
  

parameter_calc()

#Evaluate mean and standard deviation of Gaussian distribution for color R,G,B in target image
mu_r, std_r= norm.fit(r_img)
mu_b, std_b= norm.fit(b_img)
mu_g, std_g= norm.fit(g_img)


#Mean value Percentage difference of color distribution between target and predictions:
yr1=[((np.abs((mu_r - i)/mu_r))*100) for i in mu_r_pr] #Red
yb1=[((np.abs((mu_b - i)/mu_b))*100) for i in mu_b_pr]  #Blue
yg1= [((np.abs((mu_g - i)/mu_g))*100) for i in mu_g_pr]  #Green



#Std Dev Percentage difference of color distribution between target and predictions:
yr2=[((np.abs((std_r - i)/std_r))*100) for i in std_r_pr] #Red
yb2=[((np.abs((std_b - i)/std_b))*100) for i in std_b_pr]  #Blue
yg2= [((np.abs((std_g - i)/std_g))*100) for i in std_g_pr]  #Green


#Computation of arrays for Grey scale distribution:
gr_scale_img=(0.2989*np.array(r_img)+0.5870*np.array(b_img)+0.1140*np.array(g_img)) 
gr_scale_pred=(0.2989*np.array(r_pred)+0.5870*np.array(b_pred)+0.1140*np.array(g_pred)) 
y_avg1=(0.2989*np.array(yr1)+0.5870*np.array(yb1)+0.1140*np.array(yg1)) 
y_avg2=(0.2989*np.array(yr2)+0.5870*np.array(yb2)+0.1140*np.array(yg2)) 

#Definition of useful arrays:
yr51=np.zeros(file_len) #mu_r after each epoch
yb51=np.zeros(file_len) #mu_b after each epoch
yg51=np.zeros(file_len) #mu_g after each epoch
yr52=np.zeros(file_len) #std_r after each epoch
yb52=np.zeros(file_len) #std_b after each epoch
yg52=np.zeros(file_len) #std_g after each epoch
for i in range(file_len): #(fill the arrays)
  yr51[i]=mu_r
  yb51[i]=mu_b
  yg51[i]=mu_g
  yr52[i]=std_r
  yb52[i]=std_b
  yg52[i]=std_g

#Convert lists of values of mean and standard deviation of color distribution into arrays 
yr61=np.array(mu_r_pr)
yb61=np.array(mu_b_pr)
yg61=np.array(mu_g_pr)
yr71=np.array(std_r_pr)
yb71=np.array(std_b_pr)
yg71=np.array(std_g_pr)

#Plot evolution of percentage difference between target and predicted image in terms of the mean and std_dev of color distribution
def perc_diff_plotterino():
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300) 
  plt.title('Evolution of Percentage Difference from Predicted to Target img: \n train_set=10000 imgs ')
  plt.subplot(211)
  plt.errorbar(x, yr1, ecolor='red', fmt='ro', label='Red', linestyle='None', alpha=0.5)
  plt.errorbar(x, yb1, ecolor='aqua', fmt='bo', label='Blue', linestyle='None', alpha=0.5)
  plt.errorbar(x, yg1, ecolor='lime', fmt='go', label='Green', linestyle='None', alpha=0.5)
  plt.errorbar(x, y_avg1, ecolor='black', fmt='ko', linestyle='dashed', label='Grey Scale', alpha=0.5)
  plt.xlabel('# of Epoch')
  plt.ylabel('Percent')
  plt.title('Mean Percentage Difference')
  plt.legend()
  plt.subplot(212)
  plt.errorbar(x, yr2, ecolor='red', fmt='ro', label='Red', linestyle='None', alpha=0.5)
  plt.errorbar(x, yb2, ecolor='aqua', fmt='bo', label='Blue', linestyle='None', alpha=0.5)
  plt.errorbar(x, yg2, ecolor='lime', fmt='go', label='Green', linestyle='None', alpha=0.5)
  plt.errorbar(x, y_avg2, ecolor='black', fmt='ko', linestyle='dashed', label='Grey Scale', alpha=0.5)
  plt.xlabel('# of Epoch')
  plt.title('Std Dev Percentage Difference')
  plt.ylabel('Percent')
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_perc_diff_mean-std_dev_phase1_trainset_10000.jpg')
  plt.close()


#Plot evolution of color distribution for 6 chosen epochs (in  crescent order) and compare it to target image color distributions
def avr_distr_plotterino():
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300) 
  plt.subplot(211)
  plt.title('Evolution of Distribuiton of Color Intensity from Predicted to Target img: \n train_set=10000 imgs \n RGB')
  plt.hist(b_img,bins=100, histtype='bar', density=True, align='mid', label='target (Blue)', alpha = 0.5, color= 'aqua', linewidth=1.5)
  plt.hist(g_img,bins=100, histtype='bar', density=True, align='mid', label='target (Green)', alpha = 0.5, color= 'lime', linewidth=1.5)
  plt.hist(r_img,bins=100, histtype='bar', density=True, align='mid', label='target (Red)', alpha = 0.5, color= 'red', linewidth=1.5)
  for i in range(6):
      plt.hist(r_pred[(i+1)*7],bins=100, histtype='step', density=True, align='mid', label='pred.'+str((i+1)*7), color=str(0.1*(2+i)), alpha = 0.8, linewidth=1.5)
      plt.hist(b_pred[(i+1)*7],bins=100, histtype='step', density=True, align='mid', color=str(0.1*(2+i)), alpha = 0.8, linewidth=1.5)
      plt.hist(g_pred[(i+1)*7],bins=100, histtype='step', density=True, align='mid', color=str(0.1*(2+i)), alpha = 0.8, linewidth=1.5)
  plt.xlabel('Intensity of Color (scale from 0 to 1)')
  plt.ylabel('Count')
  plt.axis([0,1,0,12])
  plt.legend(loc='upper right')
  plt.subplot(212)
  plt.title('\n Grey Scale')
  plt.hist(gr_scale_img,bins=100, histtype='bar', density=True, align='mid', label='target (Grey Scale)', alpha = 0.5, color= 'grey', linewidth=1.5)
  for i in range(6):
    plt.hist(gr_scale_pred[(i+1)*7],bins=100, histtype='step', density=True, align='mid', label='pred.'+str((i+1)*7), color=str(0.1*(2+i)), alpha = 0.8, linewidth=1.5)   
  plt.xlabel('Intensity of Color (scale from 0 to 1)')
  plt.ylabel('Count')
  plt.axis([0,1,0,12])
  plt.legend(loc='upper right')
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_distr_comp_phase1_trainset_10000.jpg')
  plt.close()

#Plot evolution of absolute value of mean and std_dev of color distributions after each epoch in comparison target image  
def avr_mu_std_diff_plotterino():
  plt.figure(num=0,figsize=(12.8, 9.6), dpi=300) 
  plt.title('Evolution of Mean and Std Dev of the Distribution of RGB colors from Predicted to Target img: \n train_set=10000 imgs ')
  plt.subplot(211)
  plt.plot(x, yr51, color='red', label='Mu_Target_R', linestyle='dashed')
  plt.plot(x, yb51, color='blue', label='Mu_Target_B', linestyle='dashed')
  plt.plot(x, yg51, color='green', label='Mu_Target_G', linestyle='dashed')
  plt.errorbar(x, yb61, ecolor='aqua', fmt='bo', label='Mu_pred_B', linestyle='None', alpha=0.5)
  plt.errorbar(x, yr61, ecolor='red', fmt='ro', label='Mu_pred_R', linestyle='None', alpha=0.5)
  plt.errorbar(x, yg61, ecolor='lime', fmt='go', label='Mu_pred_G', linestyle='None', alpha=0.5)
  plt.xlabel('# of Epoch')
  plt.ylabel('Avg Abs. Value of the Mean')
  plt.legend()
  plt.subplot(212)
  plt.plot(x, yr52, color='red', label='Std_Target_R', linestyle='dashed')
  plt.plot(x, yb52, color='blue', label='Std_Target_B', linestyle='dashed')
  plt.plot(x, yg52, color='green', label='Std_Target_G', linestyle='dashed')
  plt.errorbar(x, yg71, ecolor='lime', fmt='go', label='Std_pred_G', linestyle='None', alpha=0.5)
  plt.errorbar(x, yr71, ecolor='red', fmt='ro', label='Std_pred_R', linestyle='None', alpha=0.5)
  plt.errorbar(x, yb71, ecolor='aqua', fmt='bo', label='Std_pred_B', linestyle='None', alpha=0.5)
  plt.xlabel('# of Epoch')
  plt.ylabel('Avg Abs. Value of Standard Deviation')
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/CMB_inpainting_avg_mu-std_dev_phase1_trainset_10000.jpg')
  plt.close()

perc_diff_plotterino()

avr_mu_std_diff_plotterino()

avr_distr_plotterino()

print("--- %s seconds ---" % (time.time() - start_time))

