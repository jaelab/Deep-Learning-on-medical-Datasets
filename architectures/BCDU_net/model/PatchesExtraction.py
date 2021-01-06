import random
import math
import numpy as np

class PatchesExtraction():
    def __init__(self):
        pass

    def image_centers(self, width, height, patch_width, patch_height):
        """
        Generates random values for x and y
        :param width: Width of the image
        :param height: Height of the image
        :param patch_width: Width of the patch
        :param patch_height: Height of the patch
        :return: The values for x and y
        """
        center_x = random.randint(0 + patch_width, width - patch_width)
        center_y = random.randint(0 + patch_height, height - patch_height)
        return center_x, center_y



    def divide_to_patches(self, data, data_bm, image, patch_height, patch_width, x, y):
        """

        :param data: Numpy array of the images
        :param data_bm: Numpy array of the border masks
        :param image: The image's index
        :param patch_height: The height of the patch
        :param patch_width: The width of the patch
        :param x: Randomly generated x center
        :param y: Randomly generated y center
        :return: The patch for the corresponding image and the patch for it's corresponding border mask
        """
        patch = data[image, :, y - patch_height:y + patch_height, x - patch_width:x + patch_width]
        patch_mask = data_bm[image, :, y - patch_height:y + patch_height, x - patch_width:x + patch_width]
        return patch, patch_mask

    def rand_extract_patches(self, data, data_bm, patch_width, patch_height, n_subs):
        """
        Randomly extracts n_subs patches of images
        :param data: Numpy array containing the images inputs
        :param data_bm: Numpy array with the border masks
        :param patch_width: Width of the patch
        :param patch_height: Height of the patch
        :param n_subs: Number of patches to extract out of the data
        :return: Two numpy arrays one for the patches of the images and one for the patches of the border masks
        """
        input_patches = np.empty((n_subs, data.shape[1], patch_height, patch_width))
        bm_patches = np.empty((n_subs, data_bm.shape[1], patch_height, patch_width))
        # Image dimensions
        height = data.shape[2]
        width = data.shape[3]
        n_images = data.shape[0]

        patches_per_img = int(n_subs / n_images)
        print("patches per full image: " + str(patches_per_img))

        n_iter_all = 0
        for image in range(n_images):
            n_iter_image = 0
            while n_iter_image < patches_per_img:
                half_patch_h = int(patch_height / 2)
                half_patch_w = int(patch_width / 2)
                x, y = self.image_centers(width, height, half_patch_w, half_patch_h)
                if self.check_inside(x, y, width, height, patch_height):
                    patch, patch_mask = self.divide_to_patches(data, data_bm, image, half_patch_h, half_patch_w, x, y)
                    input_patches[n_iter_all] = patch
                    bm_patches[n_iter_all] = patch_mask
                    n_iter_all += 1
                    n_iter_image += 1

                else:
                    continue
        return input_patches, bm_patches


    def check_inside(self, x_center, y_center, width, height, patch_height):
        """
        Verify that the pixel is inside the field of view
        :param x_center: x value of the pixel
        :param y_center: y value of the pixel
        :param width: Width of the image
        :param height: Height of the image
        :param patch_height: Height of the patch
        :return: Boolean value to check if the x and y values are inside the field of view
        """
        half_width = int(width / 2)
        half_height = int(height/2)
        x = x_center - half_width
        y = y_center - half_height
        patch_diagonal = int(patch_height * np.sqrt(2.0) / 2.0)
        inside = 270 - patch_diagonal
        radius = np.sqrt((x **2) + (y**2))
        if radius < inside:
            return True
        else:
            return False

    def view_patches(self, data, patch_height=64, patch_width=64, stride_height=5, stride_width=5):
        """
        Retrieve the extraced patches for the images
        :param data: Numpy array containing the images inputs
        :param patch_height: The height of the patch
        :param patch_width: The width of the patch
        :param stride_height: The height of the stride
        :param stride_width: The width of the stride
        :return: The extracted patches for images
        """
        n_images = data.shape[0]
        height = data.shape[2]
        width = data.shape[3]
        height_val = math.floor((height - patch_height) / stride_height + 1)
        width_val = math.floor((width - patch_width) / stride_width + 1)
        return self.create_patches(data, n_images, height_val, width_val, patch_height, patch_width, stride_height, stride_width)

    def create_patches(self, data, n_images, height_val, width_val, patch_height, patch_width, stride_height, stride_width):
        """
        Creates the patches for all the images
        :param data: The data to process
        :param n_images: Number of images
        :param height_val: The height of the image
        :param width_val: the width of the image
        :param patch_height: The height of the patch
        :param patch_width: The width of the patch
        :param stride_height: The height of the stride
        :param stride_width: The width of the stride
        :return: Numpay array containing all the images' patches
        """
        n_patches_per_img = height_val * width_val
        n_patches_tot = n_patches_per_img * n_images
        patches = np.empty((n_patches_tot, data.shape[1], patch_height, patch_width))
        image_idx = 0
        for image in range(n_images):
            for height_pixel in range(height_val):
                for width_pixel in range(width_val):
                    patch = data[image, :, height_pixel * stride_height:(height_pixel * stride_height) + patch_height,
                            width_pixel * stride_width:(width_pixel * stride_width) + patch_width]
                    patches[image_idx] = patch
                    image_idx += 1
        return patches

    def remove_overlap(self, data, patch_width=64,  stride_width=5):
        """

        :param data: Numpy array containing the images inputs
        :param patch_width: Width of the patch
        :param stride_width: Width of the stride
        :return: The data after removing the overlap
        """
        width = data.shape[3]
        width_leftover = (width - patch_width) % stride_width
        data = self.change_dim(data, width_leftover, width, stride_width)
        print("New images shape: \n" + str(data.shape))
        return data

    def change_dim(self, data, leftover, width, stride_width):
        """

        :param data: Numpy array containing the images inputs
        :param leftover: The image's  width leftover
        :param width: Width of the image
        :param stride_width: Width of the stride
        :return: Adjusted image dimensions
        """
        new_dim_img = None
        if (leftover != 0):
            adjusted_dim = width + (stride_width - leftover)
            new_dim_img = np.zeros((data.shape[0], data.shape[1], data.shape[2], adjusted_dim))
            new_dim_img[0:data.shape[0], 0:data.shape[1], 0:data.shape[2], 0:width] = data

        return new_dim_img