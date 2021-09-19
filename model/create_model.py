import logging
import tensorflow as tf
from model.rnn import RNN
from model.lstm import LSTM
from model.bi_rnn import BiRNN
from model.bi_lstm import BiLSTM
from util import get_mini_batch
import os


class CreateModel(object):
    def __init__(self, config, word_embedding):
        self.logger = logging.getLogger('文本分类')
        self.word_embedding = word_embedding
        self.config = config
        self.max_document_len = config["global"]["max_document_len"]
        self.class_num = config["global"]["class_num"]
        self.model_type = config["model"]["type"]
        self.fc_nums = config["model"]["fc_nums"]
        self.learning_rate = config["model"]["learning_rate"]
        self.batch_size = config["global"]["batch_size"]

        self.sess = tf.Session()
        self._build_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())


    def _build_graph(self):
        self._create_placeholder()
        self._create_embedding()
        self._create_model()
        self._build_loss()
        self._build_optimizer()
        self._evaluate_model()

    def _create_placeholder(self):
        """
        创建占位符
        :return:
        """
        self.query_placeholder = tf.placeholder(tf.int32, [None, self.max_document_len], name="query_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, name="dropout_placeholder")
        self.label_placeholder = tf.placeholder(tf.int32, [None, self.class_num], name="label_placeholder")

    # 导入词向量
    def _create_embedding(self):
        self.logger.info("开始导入词向量....")
        with tf.name_scope("word_embedding"):
            word_embedding = tf.Variable(tf.cast(self.word_embedding, dtype=tf.float32, name="word2vec"),
                                         name="word_embedding", trainable=False)
            self.input = tf.nn.embedding_lookup(word_embedding, self.query_placeholder)

    def _create_model(self):
        if self.model_type == "rnn":
            rnn = RNN(config=self.config)
            self.output = rnn.build_net(self.input)
        if self.model_type == "lstm":
            lstm = LSTM(config= self.config)
            self.output = lstm.build_net(self.input)
        if self.model_type == "birnn":
            birnn = BiRNN(config= self.config)
            self.output = birnn.build_net(self.input)
        if self.model_type == "bilstm":
            bilstm = BiLSTM(config= self.config)
            self.output = bilstm.build_net(self.input)

        with tf.name_scope("fc"):
            self.fc = tf.layers.dense(self.output, self.class_num, tf.nn.relu)

        with tf.name_scope("softmax"):
            self.logits = tf.nn.softmax(self.fc)

    def _build_loss(self):
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):
        tvars = tf.trainable_variables()
        self.grad_clip = 1.1
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))


    def _evaluate_model(self):
        """

        :return:
        """
        with tf.variable_scope("predict"):
            self.predict = tf.argmax(self.logits, 1, name="prediction", )
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predict, tf.argmax(self.label_placeholder, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy"
            )
            tf.summary.scalar("accuracy", self.accuracy)

    def train(self, train_data_set, dev_data_set=None):
        best_loss = 0
        for i in range(self.config["global"]["epoch"]):
            self.logger.info("-------------step: {}----------".format(i))
            train_batchs = get_mini_batch(train_data_set, self.batch_size)
            total_loss, total_acc = self._epoch_op(train_batchs)
            self.logger.info("step {}, avg_loss {:.4f}, avg_accuracy {:.4f}".format(i, total_loss, total_acc))
            if dev_data_set is not None:
                eval_loss, eval_acc = self.eval(dev_data_set)
                self.logger.info("step:{}----avg_dev----avg_loss {:.4f}, accuracy {:.4f}".format(i, total_loss, total_acc))
                if best_loss < eval_loss:
                    best_loss = eval_loss
                    self.saver.save(self.sess, os.path.join(self.config["model"]["path"],
                                                            self.config["model"]["name"]))

    def eval(self, data_set):
        dev_batchs = get_mini_batch(data_set, self.batch_size)
        total_loss, total_acc = self._epoch_op(dev_batchs,train_able = False)
        return total_loss, total_acc

    def _epoch_op(self, batchs, train_able=True):
        total_loss, total_acc = 0.0, 0.0
        index = 0
        for idx, batch in enumerate(batchs):
            if train_able:
                feed_dict = {
                    self.query_placeholder: batch["word_id_list"],
                    self.dropout_placeholder: self.config["model"]["dropout"],
                    self.label_placeholder: batch["label_vector"],
                }
                _,loss, acc,p1,label2 = self.sess.run([self.optimizer, self.loss, self.accuracy,self.predict, self.label_placeholder

                                            ], feed_dict)
                pass
            else:
                feed_dict = {
                    self.query_placeholder: batch["word_id_list"],
                    self.dropout_placeholder: 0,
                    self.label_placeholder: batch["label_vector"],
                }
                loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict)
            total_loss += loss
            total_acc += acc
            index += 1
            self.logger.info("batch {}, loss {:.4f}, accuracy {:.4f}".format(idx, loss, acc))
        return total_loss / index, total_acc / index
