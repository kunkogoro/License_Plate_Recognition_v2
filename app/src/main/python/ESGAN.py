import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
import base64
import io
from os.path import dirname, join
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


# Declaring Constants
IMAGE_PATH = "original.png"
SAVED_MODEL_PATH = join(dirname(__file__), "saved_model")
# SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image_path):
    """ Loads image from path and preprocesses to make it model ready
        Args:
          image_path: Path to the image file
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
      Saves unscaled Tensor Images.
      Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
      image = tf.clip_by_value(image, 0, 255)
      image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    # print("Saved as %s.jpg" % filename)


def plot_image(image, title=""):
    """
      Plots images from image tensors.
      Args:
        image: 3D image tensor. [height, width, channels].
        title: Title to display in the plot.
    """
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

# hr_image = preprocess_image(IMAGE_PATH)

# plot_image(tf.squeeze(hr_image), title="Original Image")
# save_image(tf.squeeze(hr_image), filename="Original Image")

def processing(data):
 # decode_data = base64.b64decode(data)
 # np_data = np.fromstring(decode_data,np.uint8)
 # hr_image = cv2.imdecode(np_data,cv2.IMREAD_COLOR)

    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
 # hr_image = tf.image.decode_image(tf.io.read_file(data))
 # if hr_image.shape[-1] == 4:
 #    hr_image = hr_image[...,:-1]
 # hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
 # hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
 # hr_image = tf.cast(hr_image, tf.float32)
 # hr_image = tf.expand_dims(hr_image, 0)

 hr_image = preprocess_image(data)
 plot_image(tf.squeeze(hr_image), title="Original Image")

 model = hub.load(SAVED_MODEL_PATH)
 fake_image = model(hr_image)
 fake_image = tf.squeeze(fake_image)
 # plot_image(tf.squeeze(fake_image), title="Super Resolution")

 buff = io.BytesIO()
 pil_im = Image.fromarray(tf.cast(fake_image, tf.uint8).numpy())
 pil_im.save(buff,format="PNG")
 img_str = base64.b64encode(buff.getvalue())
 # save_image(tf.squeeze(fake_image), filename="Super Resolution")
 return str(img_str,"UTF-8")
# print("Time Taken: %f" % (time.time() - start))


# save_image(tf.squeeze(fake_image), filename="Super Resolution")