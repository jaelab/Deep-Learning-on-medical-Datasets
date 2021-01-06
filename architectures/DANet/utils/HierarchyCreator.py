import pydicom
import cv2
import glob
import random
from architectures.DANet.utils.utils import *


def create_folders(root_dir):
    for location in ["train", "val", "test"]:
        for sub_folder in ["GT", "Img"]:
            os.makedirs(os.path.join(root_dir, location, sub_folder), exist_ok=True)

        if location in ["val", "test"]:
            for result_sub_folder in ["Result", "Volume/Pred", "Volume/GT"]:
                os.makedirs(os.path.join(root_dir, location, result_sub_folder), exist_ok=True)

    os.makedirs(os.path.join(root_dir, "save"), exist_ok=True)


def crop_dcm(dcm_array):
    centerX = int(dcm_array.shape[0] / 2)
    centerY = int(dcm_array.shape[1] / 2)
    newImage = dcm_array[centerX - 128:centerX + 128, centerY - 128:centerY + 128]

    return newImage


def crop_png(img):
    w, h = img.size
    left = (w - 256) / 2
    top = (h - 256) / 2
    right = (w + 256) / 2
    bottom = (h + 256) / 2

    return img.crop((left, top, right, bottom))


def get_location(i, size, split):
    if i < size * split[0]:
        return "train"
    elif size * split[0] <= i < size * (split[0] + split[1]):
        return "val"
    else:
        return "test"


def create_hierarchy(data_dir, out_dir, split=[0.75, 0.1, 0.15]):
    create_folders(out_dir)

    nb_patient = os.listdir(data_dir)
    random.shuffle(nb_patient)
    for i, no_patient in enumerate(nb_patient):
        dcm_path = os.path.join(data_dir, no_patient, "T1DUAL/DICOM_anon/InPhase")
        gt_path = os.path.join(data_dir, no_patient, "T1DUAL/Ground")

        dcm_files = glob.glob(dcm_path + "/*.dcm")
        dcm_files.sort(key=natural_keys)

        gt_files = glob.glob(gt_path + "/*.png")
        gt_files.sort(key=natural_keys)

        for j, dcm_file in enumerate(dcm_files):
            ds = pydicom.dcmread(dcm_file)
            pixel_array_numpy = ds.pixel_array.astype(float)
            img = Image.open(gt_files[j])

            # Center crop dcm
            if pixel_array_numpy.shape[0] > 256:
                pixel_array_numpy = crop_dcm(pixel_array_numpy)

            # Center crop png
            if img.size[0] > 256:
                img = crop_png(img)

            # Normalise to 0-255
            pixel_array_numpy_gray = (pixel_array_numpy - np.min(pixel_array_numpy)) / (
                        np.max(pixel_array_numpy) - np.min(pixel_array_numpy)) * 255

            name = "Subj_" + no_patient + "slice_" + str(j + 1) + ".png"
            location = get_location(i, len(nb_patient), split)
            cv2.imwrite(os.path.join(out_dir, location, "Img", name), pixel_array_numpy_gray.astype('uint8'))
            img.save(os.path.join(out_dir, location, "GT", name))

    create_3d_volume(out_dir + "/val/GT", out_dir + "/val/Volume/GT")
    create_3d_volume(out_dir + "/test/GT", out_dir + "/test/Volume/GT")
