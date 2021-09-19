import numpy as np
from collections import OrderedDict

"""
数据处理模块
"""


class ProcessData(object):
    def __init__(self, config):
        # self.stop_words = self._load_stopword(config["global"]["stop_word_file_path"])
        self.stop_words = None
        self.embedding_np, self.token2id, self.id2token = self._load_all_word_embedding(
            config["global"]["word_embedding_file_path"],
            config["global"]["word_embedding_dim"])
        self.label_ch2id = self._load_label_file(config["global"]["label_file_path"])
        self.class_num = config["global"]["class_num"]
        self.max_document_len = config["global"]["max_document_len"]

    def load_data_from_folder(self, data_path):
        """
        load data 从文件夹读取数据
        :param data_path:
        :return: [{label_ch:, query: "a b c"  }}]
        """
        data_set = []
        with open(data_path, 'r', encoding="gb18030") as fin:
            for line in fin:
                line = line.strip().split(" ")
                label = line[0]
                word_list = []
                # 先去除停用词
                if self.stop_words is not None:
                    for word in line[1:]:
                        if word not in self.stop_words:
                            word_list.append(word.lower())
                else:
                    word_list = [i.lower() for i in line[1:]]
                query = " ".join(str(i) for i in word_list)
                data_set.append({"label_ch": label, "query": query})
        return data_set

    def convert_word2id(self, data_set):
        """
        返回每个词的id
        :param data_set:
        :return:
        """
        for line in data_set:
            label_id = int(self.label_ch2id[line["label_ch"]])
            label_vector = [0 for _ in range(self.class_num)]
            label_vector[label_id] = 1
            word_id_list = [0] * self.max_document_len
            for i, word in enumerate(line["query"].split(" ")):
                if i < self.max_document_len:
                    if word in self.token2id.keys():
                        word_id_list[i] = self.token2id[word]
                else:
                    break
            line["label_id"] = label_id
            line["label_vector"] = label_vector
            line["word_id_list"] = word_id_list
        return data_set

    def _load_stopword(self, filename):
        """
        移除停用词
        :param filename:  文件名
        :return: 停用词列表
        """
        stopwords = []
        with open(filename, 'r', encoding='utf-8') as fin:
            for idx, line in enumerate(fin):
                stopwords.append(line.strip('\n'))
        return stopwords

    def _load_all_word_embedding(self, file_path, dim):
        """
        导入词向量
        :param file_path:
        :return:
        """
        embedding_list = [[0.0] * dim]
        token2id = OrderedDict({"<UNK>": 0})
        id2token = OrderedDict({0: "<UNK>"})
        with open(file_path, "r", encoding="UTF-8") as fin:
            for line in fin:
                line = line.strip().split(" ")
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                embedding_list.append(embedding)
                id = int(len(token2id))
                token2id[word] = id
                id2token[id] = word
        embedding_np = np.asarray(embedding_list)
        return embedding_np, token2id, id2token

    def _load_label_file(self, file_path):
        """
        读取label_file的文件
        :param file_path:
        :return: {"积极":0,"消极":1}
        """
        label_ch2id = {}
        with open(file_path, "r", encoding="UTF-8") as fin:
            for line in fin:
                line = line.strip().split(":")
                label_ch2id[line[1]] = line[0]
        return label_ch2id
