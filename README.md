# rnn_text_classification
use rnn lstm text classification

英文词向量路径：https://nlp.stanford.edu/projects/glove/

模型比较：
单向rnn
1 精度0.83，只用1层rnn。每个维度为128，使用梯度裁剪，批大小为64，句子长度为200，使用out = tf.reduce_mean(outputs, 1)方式计算最后一层的输出，将200个神经元的输出取均值；
2 精度0.66，3层rnn，分析主要原因是梯度消失，放弃
3 精度0.66，使用1层，使用最后一个time-step的结果输入到loss中，out = final_state[0]，结果不好

单向lstm
1 跑了9步，0.74左右

双向lstm
跑了5步，0.71