import os
import json
import pickle


def create_txt(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def create_json(filename, content):
    json_str = json.dumps(content, indent=4)
    with open(filename, 'w') as json_file:
        json_file.write(json_str)


def read_json(filename):
    assert os.path.exists(filename), print('文件{filename}不存在')
    with open(filename, "rb") as fh:
        content = json.load(fh)
    return content


def save_as_pickle(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(name):
    assert os.path.exists(name), print('文件{name}不存在')
    with open(name, "rb") as fh:
        data = pickle.load(fh)
    return data
