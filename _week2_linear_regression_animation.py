import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and an axes
fig, ax = plt.subplots()

# Initialize lists for data points
x_data, y_data = [], []

# Create a line object for the linear regression line
line, = ax.plot([], [], color='red')

# Define the true slope and intercept for the underlying linear relationship
true_slope = 2
true_intercept = 1

# Function to add normally distributed noise
def add_noise(y):
    return y + np.random.normal(0, 1, len(y))

# Function to compute linear regression parameters (slope and intercept)
def best_fit_slope_and_intercept(x, y):
    m = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) /
         ((np.mean(x)**2) - np.mean(x**2)))
    b = np.mean(y) - m * np.mean(x)
    return m, b

# Initialize a scatter plot
scatter = ax.scatter(x_data, y_data)

# Limit axes for better visualization
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)

# Title and labels
ax.set_title('Linear Regression Animation')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialization function
def init():
    line.set_data([], [])
    return line,

# Animation update function
def update(frame):
    # Add a new data point with noise
    x = np.random.rand() * 10
    y = true_slope * x + true_intercept
    y_noise = add_noise(np.array([y]))

    x_data.append(x)
    y_data.append(y_noise[0])

    # Update scatter plot data
    scatter.set_offsets(np.c_[x_data, y_data])

    # Calculate new linear regression line
    if len(x_data) > 1:
        m, b = best_fit_slope_and_intercept(np.array(x_data), np.array(y_data))
        line.set_data([0, 10], [b, m * 10 + b])

    return line, scatter

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 40), init_func=init, blit=True, interval=500)

# To save the animation, use the following line
# ani.save('linear_regression_animation.mp4', writer='ffmpeg')

plt.show()
