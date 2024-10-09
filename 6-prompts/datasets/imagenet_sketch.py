import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames

@DATASET_REGISTRY.register()
class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-sketch"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)
        # print(len(classnames))

        data = self.read_data(classnames)
        super().__init__(train_x=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        # print("the number of folders: {}".format(len(folders)))
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
