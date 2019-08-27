import os
import gc
import datetime
import numpy as np
import pandas as pd
import cv2


from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras import backend as K
from keras.utils import Sequence


import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

plt.ioff()

#import libraries from GitHub
##NOTE: this command is only for Colaboratory notebooks, generally you can just import these libraries as any other common one by doing <import 'path of the library'>##
from importlib.machinery import SourceFileLoader
from os.path import join
from os import makedirs
mask = SourceFileLoader('Mask_generator.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator.py')).load_module()
unet_100 = SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()

#Directories with training, test and validation data (remember each directory needs to contain subfolders reach repesenting the specific class of the images inside

TRAIN_DIR = r"/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_cropped_maps_center_sky/Training"
TEST_DIR = r"/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_cropped_maps_center_sky/Test"
VAL_DIR = r"/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_cropped_maps_center_sky/Validation"

BATCH_SIZE = 16

class AugmentingDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)        
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample            
            mask = np.stack([
                mask_generator.sample(seed)
                for _ in range(ori.shape[0])], axis=0
            )

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori
            

# Create training generator
train_datagen = AugmentingDataGenerator(  
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    mask.MaskGenerator(128, 128, 3),
    target_size=(128, 128), 
    batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = AugmentingDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, 
    mask.MaskGenerator(128, 128, 3), 
    target_size=(128, 128), 
    batch_size=BATCH_SIZE, 
    seed=42
)

# Create testing generator
test_datagen = AugmentingDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, 
    mask.MaskGenerator(128, 128, 3), 
    target_size=(128, 128), 
    batch_size=BATCH_SIZE, 
    seed=42
)

# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Show side by side
for i in range(len(ori)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[i,:,:,:])
    axes[1].imshow(mask[i,:,:,:] * 1.)
    axes[2].imshow(ori[i,:,:,:])
    #plt.show()


    
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_map_chunks_test_samples', exist_ok=True)


        
 # Instantiate the model (Step 1)
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')
model.load('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_ssi_phase/weights.06-2.06.h5')
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_ssi_phase', exist_ok=True)
FOLDER = '/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_ssi_phase'

# Load weights from previous run (Step 2; no Batch normalization) so check last run and its updates
#model= unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')
#model.load(
    #'/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase1/weights.10-8.18.h5',
    #train_bn=False,
    #lr=0.00005
#)
#makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2', exist_ok=True)
#makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/predict_CMB_samples', exist_ok=True)

# Run training for certain amount of epochs
model.fit_generator(
    train_generator, 
    steps_per_epoch=6250,
    epochs=5, #diregard the number of epoch, simply run step 1 until it doesn't improve anymore; then move on to step 2 and follow the same process  
    #verbose=0,
    validation_data=val_generator,
    validation_steps=625,
    callbacks=[
        TensorBoard(
            log_dir=FOLDER, #Step 1
            #log_dir='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2', #Step 2
            write_graph=False
        ),
        ModelCheckpoint(
            '/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_ssi_phase/weights.{epoch:02d}-{loss:.2f}.h5', #Step 1
            #'/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/CMB_inpainting_phase2/weights.{epoch:02d}-{loss:.2f}.h5', #Step 2
            #monitor='loss', 
            monitor='val_loss',
            save_best_only=False, 
            save_weights_only=True
        ),
        #TQDMNotebookCallback()
    ]
)
