import cv2
from datasets.RetinaBloodVesselDataset import *
from architectures.BCDU_net.model.PatchesExtraction import *
from matplotlib import pyplot as plt

class Preprocessing():
    def __init__(self):
        self.new_height = None
        self.new_width = None
        pass

    def run_preprocess_pipeline(self, data_input, type, data_bm=None):
        """
        :param data_input: The data numpy array we want to apply the preproccessing on
        :param type: The type of the dataset "train" or "test"
        :param data_bm: The data numpy array for the border masks
        :return: The preprocessed numpy arrays for the images and the border masks
        """
        gray_input = self.RGB2gray(data_input)
        self.plot_image(gray_input, "gray")
        normalized_input = self.normalize(gray_input)
        self.plot_image(normalized_input, "normalize")
        tiles_input = self.divide_to_tiles(normalized_input)
        self.plot_image(tiles_input, "clahe")
        good_gamma_input = self.fix_gamma(tiles_input, 1.2)
        self.plot_image(good_gamma_input, "gamma")
        reduced_range_input = self.reduce_range(good_gamma_input)

        if data_bm is not None:
            patches_extractor = PatchesExtraction()
            reduced_range_masks = self.reduce_range(data_bm)
            if type == "train":
                no_bottom_top_input = self.cut_bottom_top(reduced_range_input)

                no_bottom_top_bm = self.cut_bottom_top(reduced_range_masks)

                input_patches, bm_patches = patches_extractor.rand_extract_patches(no_bottom_top_input,
                                                                      no_bottom_top_bm,
                                                                      64,
                                                                      64,
                                                                      200000)
                self.plot_patches(input_patches, "extract_patches", "input")
                self.plot_patches(bm_patches, "extract_patches", "gt")

            else:
                extended_input = self.extend_images(reduced_range_input)
                bm_patches = self.extend_images(reduced_range_masks)
                removed_overlap_input = patches_extractor.remove_overlap(extended_input)
                self.new_height = removed_overlap_input.shape[2]
                self.new_width = removed_overlap_input.shape[3]
                input_patches = patches_extractor.view_patches(removed_overlap_input)
        else:
            input_patches = reduced_range_input
            bm_patches = ""
        return input_patches, bm_patches

    def plot_patches(self, data, preprocess_step="", type="input"):
        """
        Saves 10 patches of the input or groundtruth
        :param data: The preprocessed numpy array of the images or masks
        :param preprocess_step: The string specifying the current preprocess step
        :param type: The type either input or groundtruth
        """
        for i in range(10):
            plt.clf()
            plt.imshow(np.squeeze(data[i]), cmap='gray')
            plt.savefig('architectures/BCDU_net/Preprocessed_Images/Patches/' + type + preprocess_step +str(i) +"_.png")

    def plot_image(self, data, preprocess_step=""):
        """
        Saves an image representation of the data at a particular preprocessing step
        :param data: The preprocessed numpy array of the images or masks
        :param preprocess_step: The string specifying the current preprocess step
        """
        plt.clf()
        plt.imshow(np.squeeze(data[1]), cmap='gray')
        plt.savefig('architectures/BCDU_net/Preprocessed_Images/' + preprocess_step +"_.png")
        plt.clf()
        plt.imshow(np.squeeze(data[13]), cmap='gray')
        plt.savefig('architectures/BCDU_net/Preprocessed_Images/' + preprocess_step +"_13_.png")


    def new_dimensions(self):
        """

        :return: The new dimensions of the images
        """
        return self.new_height, self.new_width

    def RGB2gray(self, data):
        """

        :param data: The image's numpy array to process (RGB)
        :return: The balck and white version of the image's numpy array
        """
        channel_1 = data[:, 0, :, :]
        channel_2 = data[:, 1, :, :]
        channel_3 = data[:, 2, :, :]
        channel_1 = channel_1 * 0.299
        channel_2 = channel_2 * 0.587
        channel_3 = channel_3 * 0.114
        black_white_images = channel_1 + channel_2 + channel_3
        black_white_images = np.reshape(black_white_images,
                                        (data.shape[0],
                                         1,
                                         data.shape[2],
                                         data.shape[3]))

        return black_white_images




    def normalize(self, data):
        """

        :param data: The numpy array with all the data to process
        :return: The normalized numpy array
        """
        normalized_data = np.empty(data.shape)
        std = np.std(data)
        mean = np.mean(data)
        normalized_data = (data - mean) / std
        for image in range(data.shape[0]):
            min = np.min(normalized_data[image])
            max = np.max(normalized_data[image])
            normalized_data[image] = ((normalized_data[image] - min) / (max - min)) * 255
        return normalized_data

    def divide_to_tiles(self, data):
        """
        Dividing every image into small blocks of size 8x8
        An adaptive histogram equalization is used on each of these blocks
        :param data: The numpy array with all the data to process
        :return: The histogram equalized data
        """
        CLAHE_object = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tiles_data = np.empty(data.shape)
        for image in range(data.shape[0]):
            applied_CLAHE = CLAHE_object.apply(np.array(data[image, 0], dtype=np.uint8))
            tiles_data[image, 0] = applied_CLAHE
            img = Image.fromarray(applied_CLAHE)
            img_name = "architectures/BCDU_net/Preprocessed_Images/CLAHE/ " + str(image) + "CLAHE_image_.png"
            img.save(img_name)
        return tiles_data

    def fix_gamma(self, data, gamma=1.0):
        """
        Ajust the gamma values

        :param data: The numpy array with all the data to process
        :param gamma: Gamma value to apply the corrections with
        :return: A numpy array of the data after gamma corrections
        """
        adjusted_gammas = self.map_pixel_gamma(gamma)
        corrected_data = np.empty(data.shape)
        for image in range(data.shape[0]):
            corrected_data[image, 0] = cv2.LUT(np.array(data[image, 0], dtype=np.uint8), adjusted_gammas)

        return corrected_data

    def map_pixel_gamma(self, gamma):
        """
        Mapping every pixel value to an adjusted gamma value
        :param gamma: Gamma value to apply the corrections with
        :return: The mapping table containing the pixel value from 0 to 255 with the corrresponding gamma value
        """
        inverted_gamma = 1.0 / gamma
        fixed_gammas = []
        for pixel in range(0, 256):
            fixed_gamma = ((pixel / 255.0) ** inverted_gamma) * 255
            fixed_gammas.append(fixed_gamma)
        return np.array(fixed_gammas).astype("uint8")

    def reduce_range(self, data):
        """

        :param data: The numpy array containing the data
        :return: A numpy array with values between 0 and 1
        """
        return data / 255

    def cut_bottom_top(self, data):
        """

        :param data: The data numpy array of an image (565 x 584)
        :return: The cropped image removing the top an bottom part, array (565 x 565)
        """
        return data[:, :, 9:574, :]


    def extend_images(self, data, n_tests=20):
        """

        :param data: The data numpy array of the images
        :param n_tests: The number of test images
        :return: The formatted data numpy array
        """
        return data[0:n_tests, :, :, :]


