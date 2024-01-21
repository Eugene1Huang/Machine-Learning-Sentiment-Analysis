import torch
import torch.nn as nn
from datapre import word_to_idx


# 定义RNN模型类
class RNN(nn.Module):
    def __init__(self, embeddings, LSTM_dim, n_layers, bidirectional):
        super().__init__()

        # 使用预训练的嵌入矩阵的形状
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = False

        # 修改LSTM层的输入维度以匹配加载的参数
        self.lstm = nn.LSTM(embeddings.shape[1], LSTM_dim, num_layers=n_layers, bidirectional=bidirectional)

        # 其余部分不变
        self.fc = nn.Linear(LSTM_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_x):
        embedded = self.embedding(input_x.permute(1, 0))
        output, (hidden, cell) = self.lstm(embedded)
        output = self.dropout(hidden[-1])
        output = self.fc(output)
        output = self.sigmoid(output)

        return output


# 定义预测函数
def predict(model, input_text):
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        input_text = torch.tensor(input_text).unsqueeze(0)  # 增加一维作为批量大小
        output = model(input_text)
        prediction = (output >= 0.5).item()  # 将输出转换为二进制分类预测（0或1）
    return prediction

# 创建模型
model = RNN(torch.zeros((14412, 300)), 128, 1, False)  # 占位符，稍后会被加载的状态替换
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置模型为评估模式

# 创建一个函数来进行预测
def perform_prediction(input_text):
    input_indices = [word_to_idx[word] for word in input_text.split() if word in word_to_idx]
    if not input_indices:
        return "输入文本中的单词不在词汇表中"

    prediction = predict(model, input_indices)
    return "预测结果: 抑郁" if prediction else "预测结果: 非抑郁"

# 在Colab中使用输入框进行文本输入
input_text = input("请输入文本: ")
result = perform_prediction(input_text)
print(result)
