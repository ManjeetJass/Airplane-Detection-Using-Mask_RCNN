# fit a mask rcnn on the plane dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

import tensorflow as tf

# Assume that you have 12GB of GPU memory and want to allocate ~4GB: was 0.333
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


#  train with coco model
#  python ./plane.py --model=coco
#  train with last model
#  python ./plane.py --model=last

# class that defines and loads the plane dataset
class PlaneDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        self.add_class("dataset", 1, "plane")
        # define data locations
        if is_train:
            images_dir = dataset_dir + '/train/'
            annotations_dir = dataset_dir + '/train/'
        else:
            images_dir = dataset_dir + '/test/'
            annotations_dir = dataset_dir + '/test/'

        # find all images
        for filename in listdir(images_dir):
            # extract image id & extension
            image_id = filename[:-4]
            file_ext = filename[-3:]

            if file_ext == 'png':
                img_path = images_dir + filename
                ann_path = annotations_dir + image_id + '.xml'
                # add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('plane'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# define a configuration for the model
class PlaneConfig(Config):
    # define the name of the configuration
    NAME = "plane_cfg"
    # number of classes (background + plane)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 400


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()
    print("Model: ", args.model)

    # prepare train set
    train_set = PlaneDataset()
    train_set.load_dataset('dataset', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # prepare test/val set
    test_set = PlaneDataset()
    test_set.load_dataset('dataset', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # prepare config
    config = PlaneConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='./logs/', config=config)
    # load weights (mscoco) and exclude the output layers

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = 'mask_rcnn_plane.h5'
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    else:
        model_path = args.model

    print("Training on: " + model_path)
    model.load_weights(model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')

    print("Finished training")
    # tensorboard --logdir=~/workai/PlaneDetection/Mask_RCNN-Planes/logs/plane_cfg20200617T1349
