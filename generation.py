# 此文本用于生成，生成结果会在屏幕上直接输出。
import jieba
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding
from keras import optimizers
from keras.utils import np_utils
# 使用jieba进行分词
f = open('train_data/new_wangfeng.txt', 'r')
all_str = f.read().replace('\n', '').replace(' ', '')  # 去除空格
f.close()
cut_list = jieba.cut(all_str)
seg_list = []  # 分词后的文本数据
for c in cut_list:
    seg_list.append(c)

# 生成one-hot
vocab = sorted(list(set(seg_list)))
word_to_int = dict((w, i) for i, w in enumerate(vocab))
int_to_word = dict((i, w) for i, w in enumerate(vocab))

n_words = len(seg_list)  # 总词量
n_vocab = len(vocab)  # 词表长度
print('总词汇量：', n_words)
print('词表长度：', n_vocab)

seq_length = 100  # 句子长度
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
    seq_in = seg_list[i:i + seq_length]
    seq_out = seg_list[i + seq_length]
    dataX.append([word_to_int[word] for word in seq_in])
    dataY.append(word_to_int[seq_out])

n_simples = len(dataX)
print('样本数量：', n_simples)
X = np.reshape(dataX, (n_simples, seq_length))
y = np_utils.to_categorical(dataY)


# 网络结构
print('开始构建网络')
model = Sequential()
model.add(Embedding(n_vocab, 512, input_length=seq_length))
model.add(LSTM(512, input_shape=(seq_length, 512), return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(1024))
# model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
print('加载网络')
filename = "weights-improvement=26-0.105659.hdf5"
model.load_weights(filename)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# 生成种子
start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("Seed : ")
print(''.join([int_to_word[value] for value in pattern]))
n_generation = 400  # 生成的长度
print('开始生成，生成长度为', n_generation)
finall_result = []
for i in range(n_generation):
    x = np.reshape(pattern, (1, len(pattern)))
    prediction = model.predict(x, verbose=0)[0]
    index = np.argmax(prediction)
    result = int_to_word[index]
    # sys.stdout.write(result)
    finall_result.append(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# print(finall_result)
for i in range(len(finall_result)):
    if finall_result[i] != '。':
        print(finall_result[i], end='')
    else:
        print('。')
