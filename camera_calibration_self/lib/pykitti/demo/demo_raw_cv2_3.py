import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera_calibration_self.lib import pykitti

# Change this to the directory where you store KITTI data
basedir = 'D:/Datasets/KITTI'
date = '2011_09_26'
drive = '0001'
num_files = 108

# Load the data
dataset = pykitti.raw(basedir, date, drive, frames=range(0, num_files))

# Initialize figure for 3D plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


# Function to update figure with new point cloud
def update_fig(idx):
    # Clear previous data
    ax.clear()

    # Load and plot the next point cloud
    velo = dataset.get_velo(idx)  # Load point cloud
    ax.scatter(velo[:, 0], velo[:, 1], velo[:, 2], s=1, c=velo[:, 3], cmap='gray')
    ax.set_title(f'Velodyne scan (Frame {idx})')

    # Set axes limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-2, 2)

    plt.draw()


# Event handler for key press
def on_key(event):
    if event.key == ' ':
        global current_idx
        current_idx = (current_idx + 1) % len(dataset.frames)
        update_fig(current_idx)


# Connect the event handler and show the first point cloud
current_idx = 0
update_fig(current_idx)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
