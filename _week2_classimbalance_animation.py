import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Parameters
num_samples = 200
imbalance_factor = 0.1  # Adjust to control the level of imbalance
drift_factor = 0.02

# Generate initial data
data_imbalance = np.random.randn(num_samples, 2)

# Create class imbalance (more instances of class 1)
labels_imbalance = np.zeros(num_samples)
num_class_1 = int(imbalance_factor * num_samples)
labels_imbalance[:num_class_1] = 1

data_drift = np.random.randn(num_samples, 2)

# Create figure and axes
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Scatter plots for the initial data
sc_imbalance = axs[0].scatter(data_imbalance[:, 0], data_imbalance[:, 1], c=labels_imbalance, cmap='viridis')
sc_drift = axs[1].scatter(data_drift[:, 0], data_drift[:, 1], color='blue')

# Set titles for subplots
axs[0].set_title('Class Imbalance')
axs[1].set_title('Data Drift')

# Update function for animation
def update(frame):
    global data_imbalance, labels_imbalance, data_drift

    # Simulate dynamic class imbalance
    labels_imbalance = np.zeros(num_samples)
    num_class_1 = int(imbalance_factor * num_samples)
    labels_imbalance[:num_class_1] = 1

    # Simulate linear data drift
    data_drift[:, 0] += drift_factor * frame
    data_drift[:, 1] += drift_factor * frame

    # Update scatter plots
    sc_imbalance.set_array(labels_imbalance)
    sc_drift.set_offsets(data_drift)

# Create animation
animation = FuncAnimation(fig, update, frames=50, interval=200, repeat=False)

plt.show()

