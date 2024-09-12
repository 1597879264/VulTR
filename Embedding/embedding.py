# 燕山大学计算机科学与技术2班
# 王胜
# 开发时间：2024/3/30 20:48
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn

class CodeBERTEmbedding(nn.Module):
    def __init__(self):
        super(CodeBERTEmbedding, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("tokenizer")
        self.model = RobertaModel.from_pretrained("tokenizer")

    def forward(self, sentence):
        # 确保句子被正确地分词并截断/填充到max_length
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=20)
        outputs = self.model(**inputs)
        # 输出的维度为 [batch_size, sequence_length, hidden_size]
        return outputs.last_hidden_state

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, embeddings):
        # 由于我们期望的输入是[batch_size, sequence_length, embedding_dim]
        # 因此不需要调整维度
        lstm_out, _ = self.lstm(embeddings)
        # 输出维度为[batch_size, sequence_length, hidden_dim * 2]，因为是双向的
        return lstm_out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        # 由于BiLSTM是双向的，所以隐藏层维度需要翻倍
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

    def forward(self, lstm_out):
        # MultiHeadAttention期望的输入是[batch_size, sequence_length, hidden_dim]
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # 可以在这一步之后添加一个操作，比如取平均，来获取1*维度的向量
        # 输出维度为[batch_size, sequence_length, hidden_dim]
        return attn_output.mean(dim=1)  # 对sequence_length维度取平均，得到[batch_size, hidden_dim]

if __name__ == '__main__':
    # 示例使用
    codebert_embedding = CodeBERTEmbedding()
    bilstm = BiLSTM(embedding_dim=768, hidden_dim=128, num_layers=1)
    multi_head_attention = MultiHeadAttention(hidden_dim=256)  # 注意这里使用的是BiLSTM单向的隐藏层维度

    # 使用示例保持不变
    code = "int add(int a, int b) {return a+b;}"
    embedding = codebert_embedding(code)  # 获取嵌入[batch_size, sequence_length, hidden_size]
    print(embedding.shape)
    lstm_out = bilstm(embedding)  # lstm_out 的维度应该是 [batch_size, sequence_length, hidden_dim * 2]
    print(lstm_out.shape)
    node_features = multi_head_attention(lstm_out)
    node_features = node_features.squeeze(0)  # 移除batch_size维度, 结果维度为[hidden_dim]
    print(node_features.shape)
