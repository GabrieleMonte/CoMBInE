import os
import gc
import copy
from os import makedirs
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras_tqdm import TQDMNotebookCallback

import matplotlib
import matplotlib.pyplot as plt

import cv2
import time
from os.path import join

#import libraries from GitHub
##NOTE: this command is only for Colaboratory notebooks, generally you can just import these libraries as any other common one by doing <import 'path of the library'>##
from importlib.machinery import SourceFileLoader
mask = SourceFileLoader('Mask_generator.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/Mask_generator.py')).load_module()
unet_100= SourceFileLoader('PConv_UNet_model_100.py', join('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/', 'utils/PConv_UNet_model_100.py')).load_module()
start_time = time.time()
# Settings
BATCH_SIZE = 3

# Instantiate mask generator (generate random shaped mask on 12X128 pixel image)
mask_generator = mask.MaskGenerator(128, 128, 3, rand_seed=42)

# Load image (Disregard the path which are specific to my directory)
img = np.array(Image.open('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_map_cropped_7_x153_y143.jpg'))/255
     
# Load mask
mask = mask_generator.sample()

# Image + mask
masked_img = copy.deepcopy(img)
masked_img[mask == 0] = 1

# Show side by side
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(img)
axes[1].imshow(mask * 255)
axes[2].imshow(masked_img)
#plt.show()


def plot_sample_data(masked, mask, ori, middle_title='Raw Mask'):
    #save the plot on a different figure for every epoch (such that we can see the progress of the learning process)
    i=0
    while i in range(10):
        exists = os.path.isfile('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_1_image_predictions/predicted_sample_CMB_' + str(i + 1) + '.jpg')
        if exists:
            i+=1
        else:
            break
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[:, :, :])
    axes[0].set_title('Masked Input')
    axes[1].imshow(mask[:, :, :])
    axes[1].set_title(middle_title)
    axes[2].imshow(ori[:, :, :])
    axes[2].set_title('Target Output')
    plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_1_image_predictions/predicted_sample_CMB_'+str(i+1)+'.jpg')



class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:
            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([mask_generator.sample() for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

# Create datagen
datagen = DataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True
)

# make log folder and its recipients
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs', exist_ok=True)
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/sample_CMB_1', exist_ok=True)
makedirs('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_1_image_predictions', exist_ok=True)

# Create generator from numpy array
batch = np.stack([img for _ in range(BATCH_SIZE)], axis=0)

generator = datagen.flow(x=batch, batch_size=BATCH_SIZE)
[m1, m2], o1 = next(generator)
#plot_sample_data(m1[0], m2[0] * 255, o1[0])


# Instantiate model (initial weights are 
model = unet_100.PConvUnet(vgg_weights='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/pytorch_to_keras_vgg16.h5')

model.fit_generator(
    generator,
    #verbose=0,
    steps_per_epoch=2000,
    epochs=10,
    callbacks=[
        TensorBoard(
            log_dir='/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/sample_CMB_1',
            write_graph=False
        ),
        ModelCheckpoint(
            '/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/logs/sample_CMB_1/weights.{epoch:02d}-{loss:.2f}.h5',
            monitor='loss',
            save_best_only=False, #put True if you only want to save best weights
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_sample_data(
                masked_img,
                model.predict(
                    [
                        np.expand_dims(masked_img, 0),
                        np.expand_dims(mask, 0)
                    ]
                )[0]
                ,
                img,
                middle_title='Prediction'
            )
        ),
        #TQDMNotebookCallback()
    ],
);




#model.predict plot final results
_, new_axes = plt.subplots(1, 3, figsize=(20, 5))
new_axes[0].imshow(masked_img[:, :, :])
new_axes[0].set_title('Masked Input')
new_axes[1].imshow(model.predict([np.expand_dims(masked_img, 0), np.expand_dims(mask, 0) ])[0])
new_axes[1].set_title('Prediction')
new_axes[2].imshow(img[:, :, :])
new_axes[2].set_title('Target Output')
plt.savefig('/content/drive/My Drive/CMB_Inpainting_Oxford/CMB_Inpainting_masking/CMB_1_image_predictions/predicted_sample_CMB_final.jpg')

print("--- %s seconds ---" % (time.time() - start_time))

