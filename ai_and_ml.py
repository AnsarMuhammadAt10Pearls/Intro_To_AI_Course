import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

# Initialize data for the animation
epochs = 100
accuracy = np.zeros(epochs)
loss = np.zeros(epochs)

# Initialize loss and accuracy
loss[0] = 1
accuracy[0] = 0

# This function will be called for each frame of the animation
def update(frame):
    if frame > 0:
        accuracy[frame] = accuracy[frame - 1] + np.random.rand() * (1 - accuracy[frame - 1]) * 0.05
        loss[frame] = loss[frame - 1] * (1 - np.random.rand() * 0.1)

    ax1.clear()
    ax2.clear()

    # Update accuracy plot
    ax1.plot(accuracy[:frame], label='Accuracy', color='dodgerblue', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold', color='dodgerblue')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Update loss plot
    ax2.plot(loss[:frame], label='Loss', color='crimson', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold', color='crimson')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, linestyle='--', alpha=0.5)

# Set up the figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Adjust layout to prevent text cutting
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4)

# Creating the Animation object
ani = FuncAnimation(fig, update, frames=epochs, repeat=False)

plt.show()
