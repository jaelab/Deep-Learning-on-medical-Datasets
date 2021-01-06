from distutils.dir_util import copy_tree
import glob
import random
from architectures.DANet.utils.utils import *


def create_folders(root_dir):
    for location in ["train", "val", "test",  "save/pred"]:
        os.makedirs(os.path.join(root_dir, location), exist_ok=True)


def get_location(i, size, split):
    if i < size * split[0]:
        return "train"
    elif size * split[0] <= i < size * (split[0] + split[1]):
        return "val"
    else:
        return "test"


def create_hierarchy(data_dir, out_dir, split=[0.75, 0.1, 0.15]):
    create_folders(out_dir)

    patient_folders = glob.glob(data_dir + "/*GG/*/")
    random.shuffle(patient_folders)
    for i, patient in enumerate(patient_folders):
        loc = get_location(i, len(patient_folders), split)
        patient_name = patient.replace("\\", "/").split("/")[-2]
        copy_tree(patient, os.path.join(out_dir, loc, patient_name))


