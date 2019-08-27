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
mask2 = SourceFileLoader('Mask_generator2.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator2.py')).load_module()
unet_100 = SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()
start_time = time.time()

# Instantiate mask generators
mask_generator = mask.MaskGenerator(128, 128, 3, rand_seed=42) #random shaped mask
mask_generator2 = mask2.MaskGenerator2(128, 128, radius=25,xcenter=64,ycenter=64, channels=3, rand_seed=42) #circular mask at center with radius 25
mask_generator3 = mask2.MaskGenerator2(128, 128, radius=50,xcenter=64,ycenter=64, channels=3, rand_seed=42) #circular mask at center with radius 50

#Upload random image from validation set and generate masks
img= np.array(Image.open('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_maps_cropped_center_sky/Training/CMB_maps_cropped/CMB_map_cropped_780_x235_y105.jpg').resize((128, 128))) / 255
mask = mask_generator.sample()
mask2 = mask_generator2.sample()
mask3 = mask_generator3.sample()

# Image + masks
masked_img1 = copy.deepcopy(img)
masked_img2 = copy.deepcopy(img)
masked_img3 = copy.deepcopy(img)
masked_img1[mask == 0] = 1
masked_img2[mask2 == 0] = 1
masked_img3[mask3 == 0] = 1

#Initiate model with pytorch VGG16 image-net weights 
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')
#Load model from last trained epoch
model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/weights.19-1.02.h5')

# Run predictions for this image and the different mask
pred_img1 = model.predict([ np.expand_dims(masked_img1, 0), np.expand_dims(mask, 0)])[0]
pred_img2= model.predict([ np.expand_dims(masked_img2, 0), np.expand_dims(mask2, 0)])[0]
pred_img3= model.predict([ np.expand_dims(masked_img3, 0), np.expand_dims(mask3, 0)])[0]

#Absolute difference image
diff_img1=np.abs(pred_img1-img)
diff_img2=np.abs(pred_img2-img)
diff_img3=np.abs(pred_img3-img)

#Define and fill lists containing the values of R,G,B, color in both the target and the predicted images
r_img, g_img, b_img, r_pred1, g_pred1, b_pred1, r_pred2, g_pred2, b_pred2, r_pred3, g_pred3, b_pred3= list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
for i,j in itertools.product(range(128),range(128)):
  r_img.append(img[i][j][0])
  g_img.append(img[i][j][1])
  b_img.append(img[i][j][2])
  r_pred1.append(pred_img1[i][j][0])
  g_pred1.append(pred_img1[i][j][1])
  b_pred1.append(pred_img1[i][j][2])
  r_pred2.append(pred_img2[i][j][0])
  g_pred2.append(pred_img2[i][j][1])
  b_pred2.append(pred_img2[i][j][2])
  r_pred3.append(pred_img3[i][j][0])
  g_pred3.append(pred_img3[i][j][1])
  b_pred3.append(pred_img3[i][j][2])

#Define and fill lists containining the grey scale color for both target and predicted images 
gr_scale_img, gr_scale_pred1, gr_scale_pred2, gr_scale_pred3= list(), list(), list(), list()
gr_scale_img=(0.2989*np.array(r_img)+0.5870*np.array(b_img)+0.1140*np.array(g_img)) 
gr_scale_pred1=(0.2989*np.array(r_pred1)+0.5870*np.array(b_pred1)+0.1140*np.array(g_pred1)) 
gr_scale_pred2=(0.2989*np.array(r_pred2)+0.5870*np.array(b_pred2)+0.1140*np.array(g_pred2)) 
gr_scale_pred3=(0.2989*np.array(r_pred3)+0.5870*np.array(b_pred3)+0.1140*np.array(g_pred3)) 

#Gaussian Fits Target vs Prediction:
#R
mu_r, std_r = norm.fit(r_img)
mu_rp1, std_rp1 = norm.fit(r_pred1)
mu_rp2, std_rp2 = norm.fit(r_pred2)
mu_rp3, std_rp3 = norm.fit(r_pred3)
x = np.linspace(0, 1, 100)
p_r = norm.pdf(x, mu_r, std_r)
p_rp1 = norm.pdf(x, mu_rp1, std_rp1)
#B
mu_b, std_b = norm.fit(b_img)
mu_bp1, std_bp1 = norm.fit(b_pred1)
mu_bp2, std_bp2 = norm.fit(b_pred2)
mu_bp3, std_bp3 = norm.fit(b_pred3)
p_b = norm.pdf(x, mu_b, std_b)
p_bp1 = norm.pdf(x, mu_bp1, std_bp1)
#G
mu_g, std_g = norm.fit(g_img)
mu_gp1, std_gp1 = norm.fit(g_pred1)
mu_gp2, std_gp2 = norm.fit(g_pred2)
mu_gp3, std_gp3 = norm.fit(g_pred3)
p_g = norm.pdf(x, mu_g, std_g)
p_gp1 = norm.pdf(x, mu_gp1, std_gp1)
#Grey Scale
mu_gr_sc, std_gr_sc = norm.fit(gr_scale_img)
mu_gr_sc_p1, std_gr_sc_p1 = norm.fit(gr_scale_pred1)
mu_gr_sc_p2, std_gr_sc_p2 = norm.fit(gr_scale_pred2)
mu_gr_sc_p3, std_gr_sc_p3 = norm.fit(gr_scale_pred3) 

#Plot aside target, masked image, predicted and absolute difference image for each different mask used
def plot_target_pred_diff():
  _, (new_axes, new_axe, new_ax)= plt.subplots(3, 4, figsize=(30, 20))
  new_axes[0].imshow(masked_img1[:, :, :])
  new_axes[0].set_title('Masked Input (Random shapes)')
  new_axes[1].imshow(pred_img1[:, :, :])
  new_axes[1].set_title('Prediction')
  new_axes[2].imshow(img[:, :, :])
  new_axes[2].set_title('Target Output')
  new_axes[3].imshow(diff_img1[:, :, :])
  new_axes[3].set_title('Diff Output')
  new_axe[0].imshow(masked_img2[:, :, :])
  new_axe[0].set_title('Masked Input (Circular r=25)')
  new_axe[1].imshow(pred_img2[:, :, :])
  new_axe[1].set_title('Prediction')
  new_axe[2].imshow(img[:, :, :])
  new_axe[2].set_title('Target Output')
  new_axe[3].imshow(diff_img2[:, :, :])
  new_axe[3].set_title('Diff Output')
  new_ax[0].imshow(masked_img3[:, :, :])
  new_ax[0].set_title('Masked Input (Circular r=50)')
  new_ax[1].imshow(pred_img3[:, :, :])
  new_ax[1].set_title('Prediction')
  new_ax[2].imshow(img[:, :, :])
  new_ax[2].set_title('Target Output')
  new_ax[3].imshow(diff_img3[:, :, :])
  new_ax[3].set_title('Diff Output')
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/plot_target_pred_diff.jpg')
  plt.close()

#Plot distributions of R,G,B color and grey scale for each predicted image vs target image 
def plot_distr_target_pred_diff():
  plt.figure(figsize=(15, 12), dpi=300)
  plt.subplot(211)
  plt.hist(r_img,bins=100, histtype='bar', density=True, align='mid', label='Target (Red)', alpha = 0.5, color= 'red', linewidth=1.5)
  plt.hist(r_pred1,bins=100, histtype='step', density=True, align='mid', label='Prediction (Random)',ls='dashed', alpha = 0.8, color= 'black', linewidth=1.5)
  plt.hist(r_pred3,bins=100, histtype='step', density=True, align='mid', label='Prediction (Circular r=50)',ls='dashed', alpha = 0.8, color= 'maroon', linewidth=1.5)
  plt.hist(r_pred2,bins=100, histtype='step', density=True, align='mid', label='Prediction (Circular r=25)',ls='dashed', alpha = 0.8, color= 'olive', linewidth=1.5)
  plt.hist(b_img,bins=100, histtype='bar', align='mid', density=True, label='Target(Blue)', alpha = 0.5, color= 'aqua', linewidth=1.5)
  plt.hist(b_pred1,bins=100, histtype='step', density=True, align='mid',ls='dashed', alpha = 0.8, color= 'black', linewidth=1.5)
  plt.hist(b_pred3,bins=100, histtype='step', density=True, align='mid', ls='dashed', alpha = 0.8, color= 'maroon', linewidth=1.5)
  plt.hist(b_pred2,bins=100, histtype='step', density=True, align='mid',ls='dashed', alpha = 0.8, color= 'olive', linewidth=1.5)
  plt.hist(g_img,bins=100, histtype='bar', density=True, align='mid', label='Target(Green)', alpha = 0.5, color= 'lime', linewidth=1.5)
  plt.hist(g_pred1,bins=100, histtype='step', density=True, align='mid',ls='dashed', alpha = 0.8, color= 'black', linewidth=1.5)
  plt.hist(g_pred3,bins=100, histtype='step', density=True, align='mid',ls='dashed', alpha = 0.8, color= 'maroon', linewidth=1.5)
  plt.hist(g_pred2,bins=100, histtype='step', density=True, align='mid',ls='dashed', alpha = 0.8, color= 'olive', linewidth=1.5)
  plt.xlabel('Intensity of RGB Colors (scale from 0 to 1)')
  plt.ylabel('Count')
  plt.title('Distribuiton of RGB Color Intensity: Target vs Predicted imgs ')
  plt.axis([0,1,0,15])
  plt.legend()
  plt.subplot(212)
  plt.hist(gr_scale_img,bins=100, histtype='bar', density=True, align='mid', label='Target (Grey Scale)', alpha = 0.5, color= 'grey', linewidth=1.5)
  plt.hist(gr_scale_pred1,bins=100, histtype='step', density=True, align='mid', label='Prediction (Random)',ls='dashed', alpha = 0.8, color= 'black', linewidth=1.5)
  plt.hist(gr_scale_pred3,bins=100, histtype='step', density=True, align='mid', label='Prediction (Circular r=50)',ls='dashed', alpha = 0.8, color= 'maroon', linewidth=1.5)
  plt.hist(gr_scale_pred2,bins=100, histtype='step', density=True, align='mid', label='Prediction (Circular r=25)',ls='dashed', alpha = 0.8, color= 'olive', linewidth=1.5)
  plt.xlabel('Intensity of Grey Scale Color (scale from 0 to 1)')
  plt.ylabel('Count')
  plt.title('Distribuiton of Grey Scale Color Intensity: Target vs Predicted imgs ')
  plt.axis([0,1,0,15])
  plt.legend()
  plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_inpainting_analysis_plots/plot_distr_target_pred_diff.jpg')
  plt.close()


plot_target_pred_diff()
plot_distr_target_pred_diff()

print("--- %s seconds ---" % (time.time() - start_time))
