import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_calibration_self.lib import pykitti

# Change this to the directory where you store KITTI data
basedir = 'D:\Datasets\KITTI'
date = '2011_09_26'
drive = '0001'

# Load the data
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 20, 5))

