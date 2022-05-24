import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
import utils
import numpy as np
import shutil


def read_csv_classes(csv_dir: str, csv_name: str):
    """给出csv的地址与名称，返回该csv包含的图片名称与标签集"""
    data = pd.read_csv(os.path.join(csv_dir, csv_name))
    # print(data.head(1))  # filename, label

    label_set = set(data["label"].drop_duplicates().values)  # 去除当前csv文件中重复的label

    # print("{} have {} images and {} classes.".format(csv_name,data.shape[0],len(label_set)))
    return data, label_set


def calculate_split_info(path: str, label_dict: dict, args):
    # read all images
    image_dir = os.path.join(path, "images")
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    print("MiniImageNet find {} images in dataset.".format(len(images_list)))

    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    labels = (train_label | val_label | test_label)  # 求并集
    labels = list(labels)
    labels.sort()  # 按字符顺序上升
    print("MiniImageNet all classes: {}".format(len(labels)))

    # create classes_name.json
    # 字母数字标签到数字标签、字母标签R
    long_label_dict = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    all_class2text = dict([(v[0], [v[1]]) for k, v in long_label_dict.items()])

    # concat csv data
    data = pd.concat([train_data, val_data, test_data], axis=0)
    # print("total data shape: {}".format(data.shape))

    # 将data中的标签换为数字标签
    for k, v in long_label_dict.items():
        data.replace(k, v[1], inplace=True)

    # 提取符合要求的dataframe
    if args.extraction_order is None:  # 如果没有指定提出的类，那么就随机指定
        args.extraction_order = np.random.randint(0, 100, args.class_num)
    extracted_data = pd.DataFrame(columns=['filename', 'label'])
    for num_label in args.extraction_order:
        word_label = all_class2text[num_label][0]
        extracted_data = extracted_data.append(data[data['label'] == word_label][:args.single_class_pics])

    # 将对应的类与图片提出，放到一个新的文件夹中
    tiny_path = os.path.join(path, '..', 'tinyImageNet')
    tiny_imagePath = os.path.join(tiny_path, 'images')
    if not os.path.isdir(tiny_path):
        os.makedirs(tiny_path)
        os.makedirs(tiny_imagePath)
    extracted_data.to_csv(os.path.join(tiny_path, 'all.csv'), index=False)

    extracted_labels = list(set(extracted_data['label']))
    extracted_labels.sort()
    category_image_nums = dict(((extracted_label, 0) for extracted_label in extracted_labels))
    for idx, row in extracted_data.iterrows():
        shutil.copy(os.path.join(image_dir, row['filename']), os.path.join(tiny_imagePath, row['filename']))
        category_image_nums[row['label']]+=1

    # 生成train、test的csv文件
    split = StratifiedShuffleSplit(n_splits=1, test_size=args.test_rate, random_state=42)
    for train_index, test_index in split.split(extracted_data, extracted_data["label"]):
        strat_train_set = extracted_data.iloc[train_index, :]
        strat_test_set = extracted_data.iloc[test_index, :]  # 保证测试集
        strat_train_set.to_csv(os.path.join(tiny_path, "train.csv"), index=False)
        strat_test_set.to_csv(os.path.join(tiny_path, "test.csv"), index=False)

    dataset_att = dict()
    dataset_att['category_image_nums'] = category_image_nums
    dataset_att['dataset_pars'] = vars(args)

    utils.create_json(os.path.join(tiny_path, 'num2text_label.json'), extracted_labels)
    utils.create_json(os.path.join(tiny_path, 'dataset_att.json'), dataset_att)



def main():
    data_dir = "../mini-imagenet/"
    json_path = "./imagenet_class_index.json"

    # load imagenet labels，得到“标签-文字”字典
    label_dict = json.load(open(json_path, "r"))
    label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])

    # 输入参数，一共100类，准备分为多少类；每一类最多500张图片，准备分为多少张
    args = argparse.ArgumentParser(description='Create TinyImageNet from MiniImageNet')
    args.add_argument('-c', '--class_num', default=10, type=int,
                      help='class nums  (default is 10, should smaller than 100)')
    args.add_argument('-s', '--single_class_pics', default=600, type=int,
                      help='single class pics  (default is 600, should smaller than 600)')
    args.add_argument('-e', '--extraction_order', default=list(range(10)), type=list,
                      help='extraction order(default is [0,10), left closed right open, if it is None, random extract)')
    args.add_argument('-t', '--test_rate', default=0.2, type=float,
                      help='test rate (default is 0.2)')
    args.add_argument('-r', '--random_state', default=42, type=int,
                      help='random_state(default is 42)')
    args = args.parse_args()
    np.random.seed(args.random_state)
    assert len(args.extraction_order) == args.class_num, 'extraction order tuple error'

    calculate_split_info(data_dir, label_dict, args)


if __name__ == '__main__':
    main()
