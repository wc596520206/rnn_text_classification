# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
from process_data import ProcessData
from model.create_model import CreateModel

if __name__ == "__main__":
    # Read the input information by the user on the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config path of model",
                        default=r"config\sentiment-classification.json")

    args = parser.parse_args()
    model_file = args.config_file
    with open(model_file, "r", encoding="UTF-8") as fr:
        config = json.load(fr)

    log_path = config["global"]["log_path"]
    if log_path:
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        logger = logging.getLogger("文本分类")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="UTF-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    processData = ProcessData(config)

    logger.info("读取数据集:")
    train_data_set = processData.load_data_from_folder(config["global"]["train_data_file_path"])
    logger.info("训练集数目：{}".format(len(train_data_set)))
    train_data_set = processData.convert_word2id(train_data_set)

    dev_data_set = processData.load_data_from_folder(config["global"]["dev_data_file_path"])
    logger.info("测试集数目：{}".format(len(dev_data_set)))
    dev_data_set = processData.convert_word2id(dev_data_set)

    logger.info("开始训练+预测。。。。。")
    word_embedding = processData.embedding_np
    model = CreateModel(config=config, word_embedding=word_embedding)
    model.train(train_data_set, dev_data_set)
