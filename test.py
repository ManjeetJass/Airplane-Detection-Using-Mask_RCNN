import os
import sys
import skimage.io
import warnings

import mrcnn.model as modellib
from mrcnn import visualize

# Import config
import plane

print("Running Predict...")

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
PLANE_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_plane.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

config = plane.PlaneConfig()


class InferenceConfig(config.__class__):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-Plane
model.load_weights(PLANE_MODEL_PATH, by_name=True)

# Class names
class_names = ['BG', 'plane']

# Load image from the images folder
image = skimage.io.imread(IMAGE_DIR + '/sample.png')

# original image
skimage.io.imshow(image)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
