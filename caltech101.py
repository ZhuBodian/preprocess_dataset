"""
caltech101数据集给定的是若干文件夹，文件夹内有若干图片，这个脚本的目的在于将这些文件夹内的图片都放到一个新images文件夹中，并且生成
图片名-类别的（数字类别），总csv文件，train，test的csv文件，还有数字类别-文字类别的json文件，数据集性质的json文件
下载页面为：https://data.caltech.edu/records/20086
"""
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import utils


def main(root, test_rate):
    # 复制图片到统一文件夹并改名
    assert os.path.exists(root), f'path {root} does not exist'

    categories = sorted(os.listdir(os.path.join(root, "101_ObjectCategories")), key=str.lower)
    categories.remove("BACKGROUND_Google")  # this is not a real class

    num2text_label = dict([(idx, category) for idx, category in enumerate(categories)])
    all = pd.DataFrame(columns=['filename', 'label'])
    category_image_nums = dict()
    full_path_names = []

    for label, category in enumerate(categories):
        image_dir = os.path.join(root, '101_ObjectCategories', category)

        image_names = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
        full_names = [category + '_' + name for name in image_names]
        full_names_label = [[full_name, label] for full_name in full_names]

        all = all.append(pd.DataFrame(full_names_label, columns=['filename', 'label']))
        category_image_nums[label] = len(image_names)
        full_path_names += [os.path.join(image_dir, image_name) for image_name in image_names]

    new_root = os.path.join(root, 'my_caltech101')
    images_path = os.path.join(new_root, 'images')
    if not os.path.isdir(images_path):
        os.makedirs(new_root)
        os.makedirs(images_path)
    temp = list(all['filename'])
    for idx, full_path_name in enumerate(full_path_names):
        shutil.copy(full_path_name, os.path.join(images_path, temp[idx]))

    # 生成train、test的csv文件
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_rate, random_state=42)
    for train_index, test_index in split.split(all, all["label"]):
        strat_train_set = all.iloc[train_index, :]
        strat_test_set = all.iloc[test_index, :]  # 保证测试集
        strat_train_set.to_csv(os.path.join(new_root, "train.csv"), index=False)
        strat_test_set.to_csv(os.path.join(new_root, "test.csv"), index=False)

    all.to_csv(os.path.join(new_root, "all.csv"), index=False)
    dataset_att = dict()
    dataset_att['category_image_nums'] = category_image_nums
    dataset_att['test_rate'] = test_rate

    utils.create_json(os.path.join(new_root, 'num2text_label.json'), num2text_label)
    utils.create_json(os.path.join(new_root, 'dataset_att.json'), dataset_att)


if __name__ == '__main__':
    image_folders_path = 'E:/AAAMyCodes/myjupyter/classicStructure/data/caltech101'
    test_rate = 0.2
    main(image_folders_path, test_rate)
