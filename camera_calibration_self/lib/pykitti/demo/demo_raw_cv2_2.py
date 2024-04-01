import cv2
import numpy as np
import matplotlib.pyplot as plt
from camera_calibration_self.lib import pykitti

# Change this to the directory where you store KITTI data
basedir = 'D:\Datasets\KITTI'
date = '2011_09_26'
drive = '0001'
num_files = 108

# Load the data
dataset = pykitti.raw(basedir, date, drive, frames=range(0, num_files))

# Initialize figure
f, ax = plt.subplots(2, 2, figsize=(15, 5))

# Function to update figure
def update_fig(idx):
    first_gray = dataset.get_gray(idx)
    first_rgb = dataset.get_rgb(idx)

    # Stereo processing
    stereo = cv2.StereoBM_create()
    disp_gray = stereo.compute(np.array(first_gray[0]), np.array(first_gray[1]))
    disp_rgb = stereo.compute(
        cv2.cvtColor(np.array(first_rgb[0]), cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(np.array(first_rgb[1]), cv2.COLOR_RGB2GRAY))

    # Update plots
    ax[0, 0].imshow(first_gray[0], cmap='gray')
    ax[0, 1].imshow(disp_gray, cmap='viridis')
    ax[1, 0].imshow(first_rgb[0])
    ax[1, 1].imshow(disp_rgb, cmap='viridis')

    # Set titles
    ax[0, 0].set_title('Left Gray Image (cam0)')
    ax[0, 1].set_title('Gray Stereo Disparity')
    ax[1, 0].set_title('Left RGB Image (cam2)')
    ax[1, 1].set_title('RGB Stereo Disparity')


    plt.draw()

# Event handler for key press
def on_key(event):
    if event.key == ' ':
        global current_idx
        current_idx = (current_idx + 1) % len(dataset.frames)
        update_fig(current_idx)
        plt.show()

# Connect the event handler and show the first image
current_idx = 0
update_fig(current_idx)
f.canvas.mpl_connect('key_press_event', on_key)
plt.show()
