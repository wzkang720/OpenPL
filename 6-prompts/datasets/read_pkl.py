import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

# 定义路径
dataset_dir = "imagenet"
root = os.path.abspath(os.path.expanduser("/mnt/hdd/DATA"))
dataset_dir = os.path.join(root, dataset_dir)
image_dir = os.path.join(dataset_dir, "images")
preprocessed = os.path.join(dataset_dir, "preprocessed.pkl")
split_fewshot_dir = os.path.join(dataset_dir, "split_fewshot")
# mkdir_if_missing(split_fewshot_dir)

# 定义替换方法，创建新的 Datum 对象
def replace_impath(preprocessed_data, old_path, new_path):
    for subset in preprocessed_data.values():
        for i, datum in enumerate(subset):
            if hasattr(datum, 'impath'):
                # 创建新的 Datum 对象，修改 impath
                new_impath = datum.impath.replace(old_path, new_path)
                subset[i] = Datum(impath=new_impath, label=datum.label, classname=datum.classname)

# 读取并修改预处理的数据
if os.path.exists(preprocessed):
    with open(preprocessed, "rb") as f:
        preprocessed_data = pickle.load(f)
        # print(preprocessed_data)  # 打印原始数据以供检查

        # 替换 impath 中的路径
        replace_impath(preprocessed_data, "", "/mnt/hdd")

        # 打印替换后的结果
        # print(preprocessed_data)

    # 如果需要，你可以选择将修改后的数据重新保存到文件中
    with open(preprocessed, "wb") as f:
        pickle.dump(preprocessed_data, f)