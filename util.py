import numpy as np


def get_mini_batch(data_set, batch_size, shuffle=True):
    """
    生成每次训练的batch
    :param batch_size:
    :param shuffle:
    :return:
    """
    data_size = len(data_set)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size)[:-1]:
        batch_indices = indices[batch_start: batch_start + batch_size]
        yield one_mini_batch(data_set, batch_indices)


def one_mini_batch(data, batch_indices):
    """
    产生每一次的小的batch
    :param data:
    :param batch_indices:
    :return:
    """
    batch_data = {
        "raw_data": [data[i] for i in batch_indices],
        "word_id_list": [],
        "label_vector": []
    }
    for data in batch_data["raw_data"]:
        batch_data["word_id_list"].append(data["word_id_list"])
        batch_data["label_vector"].append(data["label_vector"])
    return batch_data
