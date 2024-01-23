import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

def generate_image(frame):
    """Generate an image with varying levels of brightness."""
    if frame % 4 == 0:
        # Very dark image
        data = np.random.random((100, 100, 3)) * 0.2
    elif frame % 4 == 1:
        # Little dark image
        #data = np.random.random((100, 100, 3)) * 0.4
        data = np.random.random((100, 100, 3)) * 0.3

    elif frame % 4 == 2:
        # Little light image
        #data = np.random.random((100, 100, 3)) * 0.6 + 0.4
        data = np.random.random((100, 100, 3)) * 0.5 + 0.3
    else:
        # Very light image
        data = np.random.random((100, 100, 3)) * 0.5 + 0.5
    return Image.fromarray((data * 255).astype('uint8'))

def predict_image(image):
    """Predict whether an image is 'very dark', 'little dark', 'little light', or 'very light'."""
    mean_brightness = np.mean(image)
    if mean_brightness < 30:
        print(mean_brightness)
        return 'Very Dark'
    elif mean_brightness < 52:
        print(mean_brightness)
        return 'Little Dark'
    elif mean_brightness < 153:
        print(mean_brightness)
        return 'Little Light'
    else:
        print(mean_brightness)
        return 'Very Light'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

def update(frame):
    img = generate_image(frame)
    ax1.clear()
    ax1.imshow(img)
    ax1.set_title("Generative AI: Creating an Image")

    ax2.clear()
    ax2.axis('off')
    prediction = predict_image(np.array(img))
    ax2.text(0.5, 0.5, f'Prediction:\n{prediction}', ha='center', va='center', fontsize=20)
    ax2.set_title("Predictive AI: Analyzing the Image")

ani = FuncAnimation(fig, update, frames=16, interval=1000)
plt.show()
