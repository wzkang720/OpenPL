import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        #if num_shots >= 1:
        seed = cfg.SEED
        preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}_shuffled-seed_{seed}.pkl")

        if os.path.exists(preprocessed):
            print(f"Loading preprocessed few-shot data from {preprocessed}")
            with open(preprocessed, "rb") as file:
                data = pickle.load(file)
                train, val, test = data["train"], data["val"] ,data["test"]
        else:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
            train, val, test = self.shuffle_labels(train, val, test)
            data = {"train": train, "val": val, "test": test}
            print(f"Saving preprocessed few-shot data to {preprocessed}")
            with open(preprocessed, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = "all"
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def shuffle_labels(*args):
        output = []
        label_new = list(set([x.label for x in args[0]]))
        random.shuffle(label_new)
        for dataset in args:
            # label_new = list(dataset.label())
            # random.shuffle(label_new)
            dataset_new = []
            for label in label_new:
                for item in dataset:
                    if item.label == label:
                        dataset_new.append(item)
            output.append(dataset_new)

        return output

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new1", "new2", "new3", "new4", "new5", "new_ratio1", "new_ratio2", "new_ratio3", "new_ratio4", "new_ratio5"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = list()
        # print(labels)
        for item in dataset:
            if item.label not in labels:
                labels.append(item.label)
        # labels = list(labels)
        # labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"base SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        elif subsample == "new1":
            selected = labels[:m + math.ceil(m/5)]
        elif subsample == "new2":
            selected = labels[:m + math.ceil(2 * m/5)]
        elif subsample == "new3":
            selected = labels[:m + math.ceil(3 * m/5)]
        elif subsample == "new4":
            selected = labels[:m + math.ceil(4 * m/5)]
        elif subsample == "new5":
            selected = labels[:]
        elif subsample == "new_ratio1":
            selected = labels[math.ceil(m/5):m + math.ceil(m/5)]
        elif subsample == "new_ratio2":
            selected = labels[math.ceil(2 * m/5):m + math.ceil(2 * m/5)]
        elif subsample == "new_ratio3":
            selected = labels[math.ceil(3 * m/5):m + math.ceil(3 * m/5)]
        elif subsample == "new_ratio4":
            selected = labels[math.ceil(4 * m/5):m + math.ceil(4 * m/5)]
        elif subsample == "new_ratio5":
            selected = labels[m:]
        #else:
            #selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    def choose_item(dataset, ratio, is_var = False):
        classname = {}
        classnames = []
        set_class = set()
        # print("len of dataset: {}".format(len(dataset)))
        for item in dataset:
            set_class.add(item.classname)
            if item.classname not in classname:
                # Assign a new label if classname is not yet mapped
                classname[item.classname] = 0
                classnames.append(item.classname)


            classname[item.classname] += 1

        # Select `1 - ratio` portion from the main dataset
        for key in classname:
            classname[key] = classname[key] * ratio
        # sampled_from_dataset = dataset[:num_from_dataset]
        sampled_from_dataset = []
        for item in dataset:
            if classname[item.classname] > 0:
                sampled_from_dataset.append(item)
                classname[item.classname] -= 1
        # print(len(set_class))
        # print(len(sampled_from_dataset))
        # print("***" * 3)
        if is_var:
            return sampled_from_dataset, classnames
        return sampled_from_dataset

    @staticmethod
    def subsample_classes_var(*args, subsample="all"):
        """Combine dataset and dataset_var according to a ratio without relabeling.

        Args:
            args: a list of datasets, e.g. train, val, and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new_ratio0", "new_ratio1", "new_ratio2", "new_ratio3", "new_ratio4", "new_ratio5"]

        # Return all classes if subsample is "all"
        if subsample == "all":
            return args

        print(f"MIX DISTRIBUTION SUBSAMPLE {subsample.upper()} CLASSES!")

        # Set the ratio based on the selected subsample
        if subsample == "new_ratio1":
            ratio = 0.2
        elif subsample == "new_ratio2":
            ratio = 0.4
        elif subsample == "new_ratio3":
            ratio = 0.6
        elif subsample == "new_ratio4":
            ratio = 0.8
        elif subsample == "new_ratio5":
            ratio = 1.0
        elif subsample == "base":
            ratio = 0.0

        output = []

        # Create a dictionary to map classnames to unique labels
        for dataset, dataset_var in zip(args[::2], args[1::2]):
            classname_to_label = {}
            current_label = 0

            # Combine dataset and dataset_var based on ratio

            sampled_from_dataset = OxfordPets.choose_item(dataset, 1 - ratio , False)
            # print(len(dataset_var))
            sampled_from_dataset_var, classnames  = OxfordPets.choose_item(dataset_var, ratio, True)
            # Select `ratio` portion from the secondary dataset
            # dataset_var_size = len(dataset_var)
            # num_from_dataset_var = math.floor(dataset_var_size * ratio)
            # sampled_from_dataset_var = dataset_var[:num_from_dataset_var]

            # Combine the two subsets
            combined_dataset= sampled_from_dataset + sampled_from_dataset_var

            imagenet_classname = []
            for item in sampled_from_dataset:
                if item.classname not in imagenet_classname:
                    imagenet_classname.append(item.classname)

            # for classname in imagenet_classname:
            #     if classname not in classnames:
            #         print(classname)
            
            # print(len(imagenet_classname)) # 998
            # print(len(classnames)) # 998
                

            dataset_new = []
            # Unify labels based on classname
            for item in combined_dataset:
                # 问题
                if item.classname in classnames:
                    if item.classname not in classname_to_label:
                    # Assign a new label if classname is not yet mapped
                        classname_to_label[item.classname] = current_label
                        current_label += 1

                # Update item label with the unified label
                    item_new = Datum(
                        impath=item.impath,
                        label=classname_to_label[item.classname],
                        classname=item.classname
                    )
                    # item.label = classname_to_label[item.classname]
                    dataset_new.append(item_new)

            output.append(dataset_new)

            # class_categories = []
            # for item in dataset_new:
            #     if item.label == 1:
            #         if item.classname not in class_categories:
            #             class_categories.append(item.classname)

            # print(len(class_categories))
            
            # combined_categories = {item.label for item in dataset_new}  # 假设每项数据是字典，包含 'category' 键
            # category_intersections = []
            # 求出原始 dataset 和组合数据集的类别交集
            # dataset_categories = {item.label for item in dataset}
            # category_intersections.append(combined_categories & dataset_categories)  # 求交集

            # print(category_intersections)

        return output
        
    def balance_item(data_var, data_image, labels_var, labels, relabels):
        num_class = {}

        for item in data_var:
            if item.classname not in num_class:
                num_class[item.classname] = 1
            else:
                num_class[item.classname] += 1
        
        min_class = min(num_class.values())
        print(min_class)
        cnt_class = {}
        new_data = []
        for item in data_image:
            if item.classname not in cnt_class:
                cnt_class[item.classname] = 1
            else:
                cnt_class[item.classname] += 1
            if item.classname in labels and cnt_class[item.classname] <= min_class:
                new_data.append(
                    Datum(
                        impath=item.impath,
                        label=relabels[item.classname],
                        classname=item.classname
                    )
                )
        # print(cnt_class)
        cnt_class.clear()
        # print("cnt class:", len(cnt_class))
        for item in data_var:
            if item.classname not in cnt_class:
                cnt_class[item.classname] = 1
            else:
                cnt_class[item.classname] += 1
            if item.classname in labels_var and cnt_class[item.classname] <= min_class:
                new_data.append(
                    Datum(
                        impath=item.impath,
                        label=relabels[item.classname],
                        classname=item.classname
                    )
                )
        # print(cnt_class)
        print(len(new_data))
        return new_data
            
        
    @staticmethod
    def subsample_classes_cross_dataset(*args, subsample="all"):
        """Combine dataset and dataset_var according to a ratio without relabeling.

        Args:
            args: a list of datasets, e.g. train, val, and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new_ratio0", "new_ratio1", "new_ratio2", "new_ratio3", "new_ratio4", "new_ratio5"]

        # Return all classes if subsample is "all"
        if subsample == "all":
            return args

        print(f"CROSS DATASET SUBSAMPLE {subsample.upper()} CLASSES!")

        # Set the ratio based on the selected subsample
        if subsample == "new_ratio1":
            ratio = 0.2
        elif subsample == "new_ratio2":
            ratio = 0.4
        elif subsample == "new_ratio3":
            ratio = 0.6
        elif subsample == "new_ratio4":
            ratio = 0.8
        elif subsample == "new_ratio5":
            ratio = 1.0
        elif subsample == "base":
            ratio = 0.0

        output = []

        # Create a dictionary to map classnames to unique labels
        for dataset, dataset_var in zip(args[::2], args[1::2]):
        
            labels = list()
            labels_var = list()
            # print(labels)

            for item in dataset_var:
                if item.classname not in labels_var:
                    labels_var.append(item.classname)
            
            for item in dataset:
                # 保证不相交
                if item.classname not in labels and item.classname not in labels_var:
                    labels.append(item.classname)
            # print(len(labels) , len(labels_var)) 988 50
            n = len(labels)
            n_var = len(labels_var)
            # for every seed run, the shuffle is different, but for the same seed, the result is the same
            # print(labels)
            # random.shuffle(labels)
            # print(labels)
            labels = labels[:math.ceil(n_var * (1 - ratio))]
            labels_var = labels_var[:math.ceil(n_var * ratio)]

            # print("ratio image:", math.ceil(n_var * (1 - ratio)))
            # print("ratio var:", math.ceil(n_var * ratio))

            selected = labels + labels_var
            relabeler = {y: y_new for y_new, y in enumerate(selected)}

            # Combine dataset and dataset_var based on ratio
            combined_dataset  = OxfordPets.balance_item(dataset_var, dataset, labels_var, labels, relabeler)

            output.append(combined_dataset)

        return output