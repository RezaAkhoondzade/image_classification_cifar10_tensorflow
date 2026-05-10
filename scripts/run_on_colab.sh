"""
This is amazingly works but only needs to first copy CIFAR data on specific folder
"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/image_classification_tensorflow_project')

!mkdir /root/.keras/datasets/
!cp /content/drive/MyDrive/cifar_data/cifar-10-batches-py-target_archive /root/.keras/datasets/

!python main.py
