import numpy as np
import math

from architectures.BCDU_net.model.Preprocessing import Preprocessing


class VisualizePredicitons():
    def __init__(self):
        self.stride_weight = 5
        self.stride_height = 5
        pass


    def build_images(self, predictions, height, width):
        """
        Rebuild the prediction images to view them
        :param predictions: Numpy array with our predictions
        :param height: Height of the original image
        :param width: Width of the original image
        :return: Numpy array with the formatted predictions
        """
        n_preds = predictions.shape[0]
        patch_height = predictions.shape[2]
        patch_width = predictions.shape[3]
        n_patches_in_height = math.floor((height - patch_height) / self.stride_height + 1)
        n_patches_in_width = math.floor((width - patch_width) / self.stride_weight + 1)
        n_patches = n_patches_in_height * n_patches_in_width
        all_images = math.floor(n_preds/n_patches)
        sum_probas, sum = self.patches_sum_probas(all_images, predictions, height, width, patch_height, patch_width, n_patches_in_height, n_patches_in_width)
        formatted_predictions = sum_probas/sum
        print(formatted_predictions.shape)
        return formatted_predictions

    def patches_sum_probas(self, all_images, predictions, height, width, patch_height, patch_width, n_patches_in_height, n_patches_in_width):
        """

        :param all_images: Number of images in total
        :param predictions: Numpy array with our predictions
        :param height: Height of the image
        :param width: Width of the image
        :param patch_height: Height of the patch
        :param patch_width: Width of the patch
        :param n_patches_in_height: Number of patches in the image's height
        :param n_patches_in_width: Number of patches in the image's width
        :return: Sum of the probabilities
        """
        sum_probas = np.zeros((all_images, predictions.shape[1], height, width))
        sum = np.zeros((all_images, predictions.shape[1], height, width))
        patches_iter = 0
        for image in range(all_images):
            for i_height in range(n_patches_in_height):
                for j_width in range(n_patches_in_width):
                    tmp_dim = i_height * self.stride_height
                    tmp_dim2 = j_width * self.stride_weight
                    sum_probas[image, :, tmp_dim:(tmp_dim) + patch_height, tmp_dim2:(tmp_dim2) + patch_width] += predictions[patches_iter]
                    sum[image, :, tmp_dim:(tmp_dim) + patch_height, tmp_dim2:(tmp_dim2) + patch_width] += 1
                    patches_iter += 1
        return sum_probas, sum

    def set_black_border(self, predictions, border_masks):
        """
        Creates a black border around the image's field of view
        :param predictions: Numpy array with our predictions
        :param border_masks: Numpy array of the border masks
        """
        height = predictions.shape[2]
        width = predictions.shape[3]
        for image in range(predictions.shape[0]):
            for pixel_width in range(width):
                for pixel_height in range(height):
                    if self.in_view(image, pixel_width, pixel_height, border_masks) == False:
                        predictions[image, :, pixel_height, pixel_width] = 0.0



    def in_view(self, image, width_pixel, height_pixel, original_bm):
        """
        Check if the predictions are inside the field of view
        :param image: Image's index
        :param width_pixel: Pixel's position on the x axis
        :param height_pixel: Pixel's position on the y axis
        :param original_bm: Original border mask
        :return: Boolean value if the value is outside the border mask
        """

        orig_height = original_bm.shape[2]
        orig_width = original_bm.shape[3]

        if (width_pixel >= orig_width or height_pixel >= orig_height):
            return False

        if (original_bm[image, 0, height_pixel, width_pixel] > 0):
            return True
        else:
            return False

    def set_original_dimensions(self, data, test_inputs):
        """

        :param data: A numpy array with the data
        :param test_inputs: The test dataset inputs
        :return: The data numpy array with the orginal dimensions
        """
        height = test_inputs.shape[2]
        width = test_inputs.shape[3]
        return data[:,:,0:height,0:width]

    def field_of_view(self, images, masks, original_bm):
        """
        Returns only the predictions inside the field of view for the images and the border masks
        :param images: Numpy array with the prediction images
        :param masks: Numpy array with the border masks (groundtruth)
        :param original_bm:
        :return: Two numpy arrays, one for the predictions of the pixels inside the field of view and one for the border masks
        """
        height = images.shape[2]
        width = images.shape[3]
        new_preds = []
        new_masks = []
        for image in range(images.shape[0]):
            for width_pixel in range(width):
                for height_pixel in range(height):
                    if self.in_view(image, width_pixel, height_pixel, original_bm):
                        new_preds.append(images[image, :, height_pixel, width_pixel])
                        new_masks.append(masks[image, :, height_pixel, width_pixel])
        return np.asarray(new_preds), np.asarray(new_masks)

    def make_visualizable(self, predictions, new_h, new_w, evaluation, test_prepro_bm):
        """
        Apply all the modifications to visualize the predictions
        :param predictions:  Numpy array with our predictions
        :param new_h: New height
        :param new_w:  New width
        :param evaluation: Instance of the Evaluation class
        :param test_prepro_bm: The testing broder masks
        :return: Three numpy arrays, one for the images, the predictions and grountruth
        """
        prediction_images = self.build_images(predictions, new_h, new_w)
        prepro = Preprocessing()
        images, _ = prepro.run_preprocess_pipeline(evaluation.test_inputs[0:prediction_images.shape[0], :, :, :], "test")
        self.set_black_border(prediction_images, evaluation.test_bm)
        images = self.set_original_dimensions(images, evaluation.test_inputs)
        predictions_images = self.set_original_dimensions(prediction_images, evaluation.test_inputs)
        gt = self.set_original_dimensions(test_prepro_bm, evaluation.test_inputs)
        return images, predictions_images, gt
