import numpy as np
import os
from PIL import Image

class RetinaBloodVesselDataset():

    def __init__(self,
                 training_images_path="rawdata/DRIVE/training/images/",
                 training_groundTruth_path="rawdata/DRIVE/training/1st_manual/",
                 training_borderMasks_path = "rawdata/DRIVE/training/mask/",
                 testing_images_path="rawdata/DRIVE/test/images/",
                 testing_groundTruth_path="rawdata/DRIVE/test/1st_manual/",
                 testing_borderMasks_path="rawdata/DRIVE/test/mask/",
                 n_images=20,
                 n_channels=3,
                 height=584,
                 width=565
                 ):
        self.training_images_path = training_images_path
        self.training_groundTruth_path = training_groundTruth_path
        self.training_borderMasks_path = training_borderMasks_path
        self.testing_images_path = testing_images_path
        self.testing_groundTruth_path = testing_groundTruth_path
        self.testing_borderMasks_path = testing_borderMasks_path
        self.n_images = n_images
        self.n_channels = n_channels
        self.height = height
        self.width = width


    def extract_datasets(self, inputs_path , ground_truth_path, border_masks_path, dataset_type = ""):
        """

        :param inputs_path: Path to the file containing the training images
        :param ground_truth_path: Path to the file containing the training images' groundtruth
        :param border_masks_path: Path to the file containing the training images' border masks
        :param dataset_type: The type of the dataset "train" or "test"
        :return: Three numpy arrays, one for the training images, groundtruth and border masks
        """
        inputs = np.empty((self.n_images, self.height, self.width, self.n_channels))
        groundTruth_values = np.empty((self.n_images, self.height, self.width))
        border_masks_values = np.empty((self.n_images, self.height, self.width))
        for path, sub_directory, files in os.walk(inputs_path):
            for file in range(len(files)):
                # original
                print("original image: " + files[file])
                inputs[file] = self.convert_img_to_array(inputs_path, files[file])
                image_gT_file = files[file][0:2] + "_manual1.gif"
                print("ground truth name: " + image_gT_file)
                groundTruth_values[file] = self.convert_img_to_array(ground_truth_path, image_gT_file)
                image_bM_file = self.get_corresponding_bm(files[file], dataset_type)

                print("border masks name: " + image_bM_file)
                border_masks_values[file] = self.convert_img_to_array(border_masks_path, image_bM_file)
        return self.reshape_inputs(inputs), \
               self.reshape_gt_bm(groundTruth_values), \
               self.reshape_gt_bm(border_masks_values)

    def convert_img_to_array(self, path, file_name):
        """

        :param path: Path to the corresponding image file
        :param file_name: Name of the image file
        :return: Array of the image
        """
        complete_path = path + file_name
        image = Image.open(complete_path)
        return np.asarray(image)

    def get_corresponding_bm(self, file, dataset_type):
        """

        :param file: The name of the image file ex : 21_training.tif
        :param dataset_type: The type of the dataset "train" or "test"
        :return:  The corresponding border mask for the image
        """
        if dataset_type == "train":
            image_bM_file = file[0:2] + "_training_mask.gif"
        elif dataset_type == "test":
            image_bM_file = file[0:2] + "_test_mask.gif"
        else:
            print("dataset_type must be train or test")
            exit()
        return image_bM_file

    def reshape_inputs(self, inputs):
        """

        :param inputs: The array to reshape
        :return: The reshaped array after applying a transpose operation
        """
        return np.transpose(inputs, (0,3,1,2))

    def reshape_gt_bm(self, data):
        """

        :param ground_truth: The numpy array for the images' groundtruth or border masks
        :return: The reshaped numpy array of the groundtruth or the border masks
        """
        return np.reshape(data,(self.n_images, 1, self.height, self.width))



    def get_training_data(self):
        """

        :return: The three arrays of the training data corresponding to the images, the groundtruth and the border masks
        """
        return self.extract_datasets(self.training_images_path,
                                     self.training_groundTruth_path,
                                     self.training_borderMasks_path, "train")
    def get_testing_data(self):
        """

        :return: The three arrays of the test data corresponding to the images, the groundtruth and the border masks
        """
        return self.extract_datasets(self.testing_images_path,
                                     self.testing_groundTruth_path,
                                     self.testing_borderMasks_path, "test")



