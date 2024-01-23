import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    ax1.plot(accuracy[:frame], label='Accuracy', color='green')
    ax1.set_title('Model Accuracy Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Update loss plot
    ax2.plot(loss[:frame], label='Loss', color='red')
    ax2.set_title('Model Loss Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 1)
    ax2.legend()

# Set up the figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Creating the Animation object
ani = FuncAnimation(fig, update, frames=epochs, repeat=False)

plt.tight_layout()
plt.show()
