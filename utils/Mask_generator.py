import os
from random import randint, seed
import itertools
import numpy as np
import cv2

from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D


class MaskGenerator():

    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if
                               any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

            # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        return 1 - img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        """Loads a mask from disk, and optionally augments it"""

        # Read image
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(self.mask_files, 1, replace=False)[0]))

        # Random rotation
        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Random dilation
        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Random cropping
        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y + self.height, x:x + self.width]

        return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        """Retrieve a random mask"""
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()


class ImageChunker(object):

    def __init__(self, rows, cols, overlap):
        self.rows = rows
        self.cols = cols
        self.overlap = overlap

    def perform_chunking(self, img_size, chunk_size):
        """
        Given an image dimension img_size, return list of (start, stop)
        tuples to perform chunking of chunk_size
        """
        chunks, i = [], 0
        while True:
            chunks.append((i * (chunk_size - self.overlap / 2), i * (chunk_size - self.overlap / 2) + chunk_size))
            i += 1
            if chunks[-1][1] > img_size:
                break
        n_count = len(chunks)
        chunks[-1] = tuple(x - (n_count * chunk_size - img_size - (n_count - 1) * self.overlap / 2) for x in chunks[-1])
        chunks = [(int(x), int(y)) for x, y in chunks]
        return chunks

    def get_chunks(self, img, scale=1):
        """
        Get width and height lists of (start, stop) tuples for chunking of img.
        """
        x_chunks, y_chunks = [(0, self.rows)], [(0, self.cols)]
        if img.shape[0] > self.rows:
            x_chunks = self.perform_chunking(img.shape[0], self.rows)
        else:
            x_chunks = [(0, img.shape[0])]
        if img.shape[1] > self.cols:
            y_chunks = self.perform_chunking(img.shape[1], self.cols)
        else:
            y_chunks = [(0, img.shape[1])]
        return x_chunks, y_chunks

    def dimension_preprocess(self, img, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        this function is used to add padding and chunk up the image into pieces
        of 512x512, which can then later be reconstructed into the original image
        using the dimension_postprocess() function.
        """

        # Assert single image input
        assert len(img.shape) == 3, "Image dimension expected to be (H, W, C)"

        # Check if we are adding padding for too small images
        if padding:

            # Check if height is too small
            if img.shape[0] < self.rows:
                padding = np.ones((self.rows - img.shape[0], img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=0)

            # Check if width is too small
            if img.shape[1] < self.cols:
                padding = np.ones((img.shape[0], self.cols - img.shape[1], img.shape[2]))
                img = np.concatenate((img, padding), axis=1)

        # Get chunking of the image
        x_chunks, y_chunks = self.get_chunks(img)

        # Chunk up the image
        images = []
        for x in x_chunks:
            for y in y_chunks:
                images.append(
                    img[x[0]:x[1], y[0]:y[1], :]
                )
        images = np.array(images)
        return images

    def dimension_postprocess(self, chunked_images, original_image, scale=1, padding=True):
        """
        In case of prediction on image of different size than 512x512,
        the dimension_preprocess  function is used to add padding and chunk
        up the image into pieces of 512x512, and this function is used to
        reconstruct these pieces into the original image.
        """

        # Assert input dimensions
        assert len(original_image.shape) == 3, "Image dimension expected to be (H, W, C)"
        assert len(chunked_images.shape) == 4, "Chunked images dimension expected to be (B, H, W, C)"

        # Check if we are adding padding for too small images
        if padding:

            # Check if height is too small
            if original_image.shape[0] < self.rows:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[0:scale * original_image.shape[0], :, :])
                chunked_images = np.array(new_images)

            # Check if width is too small
            if original_image.shape[1] < self.cols:
                new_images = []
                for img in chunked_images:
                    new_images.append(img[:, 0:scale * original_image.shape[1], :])
                chunked_images = np.array(new_images)

        # Put reconstruction into this array
        new_shape = (
            original_image.shape[0] * scale,
            original_image.shape[1] * scale,
            original_image.shape[2]
        )
        reconstruction = np.zeros(new_shape)

        # Get the chunks for this image
        x_chunks, y_chunks = self.get_chunks(original_image)

        i = 0
        s = scale
        for x in x_chunks:
            for y in y_chunks:
                prior_fill = reconstruction != 0
                chunk = np.zeros(new_shape)
                chunk[x[0] * s:x[1] * s, y[0] * s:y[1] * s, :] += chunked_images[i]
                chunk_fill = chunk != 0

                reconstruction += chunk
                reconstruction[prior_fill & chunk_fill] = reconstruction[prior_fill & chunk_fill] / 2

                i += 1

        return reconstruction


class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):
        """Adapted from original _Conv() layer of Keras
        param input_shape: list of dimensions for [img, mask]
        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
            (int((self.kernel_size[0] - 1) / 2), int((self.kernel_size[0] - 1) / 2)),
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        '''

        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                'PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images * masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        return [img_output, mask_output]

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]


