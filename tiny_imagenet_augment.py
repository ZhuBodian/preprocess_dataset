import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
import utils
import numpy as np
import shutil
from torchvision import datasets, transforms
from PIL import Image


def calculate_split_info(path, args):
    original_image_dir = os.path.join(path, "images")
    num2text_label = utils.read_json(os.path.join(path, 'num2text_label.json'))
    dataset_att = utils.read_json(os.path.join(path, 'dataset_att.json'))
    nums = args.no_less_than
    augment_tinyImageNet_path = os.path.join(path, '../', 'tinyImageNetAugment')
    augment_image_dir = os.path.join(augment_tinyImageNet_path, 'images')
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    num2text_label = utils.read_json(os.path.join(path, 'num2text_label.json'))
    class_num_names_list = [k for k, _ in num2text_label.items()]

    if not os.path.isdir(augment_tinyImageNet_path):
        os.makedirs(augment_tinyImageNet_path)
        os.makedirs(augment_image_dir)

    for idx, row in train_df.iterrows():
        shutil.copy(os.path.join(original_image_dir, row['filename']), os.path.join(augment_image_dir, row['filename']))

    for idx, row in test_df.iterrows():
        shutil.copy(os.path.join(original_image_dir, row['filename']), os.path.join(augment_image_dir, row['filename']))

    train_rate = 1-dataset_att['dataset_pars']['test_rate']
    # 10类，初始每类600，测试集0.2，最终要每类2000，总的图片数：
    # （2000-600*0.8-600*0.2+600*1）*10
    orinal_and_augment_nums = dict((k, [v, max(0, int(nums - v * train_rate))])for k, v in dataset_att['category_image_nums'].items())
    additional_name_list = []
    additional_target_list = []

    all_train_label = np.array(train_df['label'])

    print(f'There are {len(class_num_names_list)} classes'.center(100, '*'))
    for class_num_name in class_num_names_list:
        print(f'Class {int(class_num_name)}'.center(50, '*'))

        additional_nums = orinal_and_augment_nums[class_num_name][1]
        class_idx = np.where(all_train_label == int(class_num_name))[0]
        rand_idxs = np.random.randint(0, len(class_idx), additional_nums)
        additional_idxs = np.array([class_idx[idx] for idx in rand_idxs])
        augment_times = dict()
        for idx in additional_idxs:
            if idx in augment_times.keys():
                augment_times[idx]+=1
            else:
                augment_times[idx] = 1

        trsfm = transforms.Compose([transforms.RandAugment()])

        for original_data_idx, times in augment_times.items():
            for i in range(times):
                image_path = os.path.join(os.getcwd(), original_image_dir, train_df.iloc[original_data_idx]['filename'])
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                temp = trsfm(image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
                # image.show()
                # temp.show()

                additional_name = train_df.iloc[original_data_idx]['filename'].split('.')[0] + '_augment_' + str(i) + '.jpg'
                additional_name_list.append(additional_name)
                additional_target_list.append(train_df.iloc[original_data_idx]['label'])
                temp.save(os.path.join(augment_image_dir, additional_name))

    additional_df = pd.DataFrame(list(map(list,zip(*[additional_name_list, additional_target_list]))), columns=['filename', 'label'])

    train_df = pd.concat([train_df, additional_df], ignore_index=True)
    dataset_att['every_class_no_less_than'] = nums
    dataset_att['orinal_and_augment_nums'] = orinal_and_augment_nums

    utils.create_json(os.path.join(augment_tinyImageNet_path, 'num2text_label.json'), num2text_label)
    utils.create_json(os.path.join(augment_tinyImageNet_path, 'dataset_att.json'), dataset_att)
    train_df.to_csv(os.path.join(augment_tinyImageNet_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(augment_tinyImageNet_path, 'test.csv'), index=False)


def main():
    # 需要先运行完毕tiny_imageNet
    data_dir = "../tinyImageNet/"  # tinyImageNet地址

    # 输入参数，一共100类，准备分为多少类；每一类最多500张图片，准备分为多少张
    args = argparse.ArgumentParser(description='Create TinyImageNet from MiniImageNet')
    args.add_argument('-n', '--no_less_than', default=1500, type=int,
                      help='every class have no less than this number')
    args.add_argument('-r', '--random_state', default=42, type=int,
                      help='random_state(default is 42)')
    args = args.parse_args()
    np.random.seed(args.random_state)

    calculate_split_info(data_dir, args)


if __name__ == '__main__':
    main()
