import numpy as np
import pandas as pd
# 从CSV文件中读取数据
df_depression = pd.read_csv('./dist/depression_dataset_reddit_cleaned.csv')
df_depression.head()

# 提取文本数据和标签，并随机打乱数据
data = [x for x in df_depression['clean_text']]
labels = [x for x in df_depression['is_depression']]
# data, labels = shuffle(data, labels)

# 加载嵌入（Embeddings）文本
embeddings_text = open('./dist/glove.6B.300d.txt', 'r', encoding='utf-8')

# 统计数据集中的单词
words = {}
for example in data:
    for word in example.split():
        if word not in words:
            words[word] = 0
        else:
            words[word] += 1

# 读取嵌入文本中的单词和对应的嵌入向量
embs = {}
for line in embeddings_text:
    split = line.split()
    word = split[0]
    if word in words:
        try:
            embedding = np.array([float(value) for value in split[1:]])
            embs[word] = embedding
        except:
            print('error loading embedding')


# 创建嵌入矩阵和单词索引
# 初始化一个空列表，用于存储词嵌入矩阵
embedding_matrix = []
# 初始化一个空列表，用于存储索引对应的词
idx_to_word = []
# 初始化一个空字典，用于存储词对应的索引
word_to_idx = {}
# 在词嵌入矩阵的第一行添加一个全零向量，用于神经网络的零填充
embedding_matrix.append(np.zeros(300))
# 在索引对应的词的列表中添加一个空字符串，表示零填充
idx_to_word.append('')
# 在词对应的索引的字典中添加一个空字符串，其索引为0
word_to_idx[''] = 0
# 遍历embs字典，它包含了词和对应的词嵌入向量
for i, (word, emb) in enumerate(embs.items()):
    # 在词嵌入矩阵中添加词的词嵌入向量
    embedding_matrix.append(emb)
    # 在索引对应的词的列表中添加词
    idx_to_word.append(word)
    # 在词对应的索引的字典中添加词，其索引为i+1
    word_to_idx[word] = i + 1
# 将词嵌入矩阵转换为numpy数组，方便后续的计算
embedding_matrix = np.asarray(embedding_matrix)

# 数据预处理和填充
# 初始化一个空列表，用于存储索引序列
x_train = []
# 遍历数据中的每个文本
for example in data:
    # 初始化一个空列表，用于存储文本中的词的索引
    temp = []
    # 将文本按空格分割为词
    for word in example.split():
        # 如果词在词对应的索引的字典中，将其索引添加到列表中
        if word in word_to_idx:
            temp.append(word_to_idx[word])
    # 将列表添加到索引序列中
    x_train.append(temp)
# 将索引序列转换为numpy数组，方便后续的计算
x_train = np.asarray(x_train)
# 遍历索引序列中的每个列表
for i in range(len(x_train)):
    # 如果列表的长度大于200，将其截断为前200个元素
    x_train[i] = x_train[i][:200]
# 遍历索引序列中的每个列表
for i in range(len(x_train)):
    # 如果列表的长度小于200，将其在前面填充零，使其长度为200
    x_train[i] = np.pad(x_train[i], (200 - len(x_train[i]), 0), 'constant')
# 初始化一个空列表，用于存储最终的索引序列
x_train_data = []
# 遍历索引序列中的每个列表
for x in x_train:
    # 将列表中的元素添加到最终的索引序列中
    x_train_data.append([k for k in x])
# 将最终的索引序列转换为numpy数组，方便后续的计算
x_train_data = np.array(x_train_data)
