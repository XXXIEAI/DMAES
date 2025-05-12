import torch
import torch.nn as nn
import torch.nn.functional as F

'''相似度计算'''
class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(SiameseNetwork, self).__init__()
        self.embed_dim = embed_dim

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def forward(self, attn_weights, X, Prompt_Essay_Embedding, num_classes):
        '''
        attn_weights: batch, prompt_size
        X: batch, embed_dim
        Prompt_Essay_Embedding: 8, embed_dim
        '''
        x1 = self.fc(X).unsqueeze(1)  # (batch_size, 1, 64)
        x2 = self.fc(Prompt_Essay_Embedding).unsqueeze(0)  # (1, 8, 64)
        distance = torch.norm(x1 - x2, p=2, dim=2) 
        # print(distance.shape)  # (batch_size, 8)

        attn_weights = attn_weights.unsqueeze(2)  # (batch_size, 2, 1)
        # print(attn_weights.shape)  # (batch_size, 2, 1)
        attn_weights_expanded = attn_weights.expand(-1, -1, num_classes)
        # print(attn_weights_expanded.shape)  # (batch_size, 2, 4))

        weight_tensor = torch.ones_like(distance)  # (batch_size, 8)
        weight_tensor[:, :num_classes] *= attn_weights_expanded[:, 0, :]
        weight_tensor[:, :num_classes] *= attn_weights_expanded[:, 1, :]
        # print(weight_tensor)

        weighted_distance = distance * weight_tensor  # 逐元素乘法
        return weighted_distance


# embed_dim = 768
# batch_size = 4
# reference_num = 8

# model = SiameseNetwork(embed_dim)

# X = torch.randn(batch_size, embed_dim)  # (4, 768)
# Prompt_Essay_Embedding = torch.randn(reference_num, embed_dim)  # (8, 768)
# attn_weights = torch.tensor([0.5, 1.5])  # 示例权重

# distance = model(attn_weights, X, Prompt_Essay_Embedding)

# print("weighted_distance.shape:", distance.shape)  # (4, 8)
# print("weighted_distance:", distance)


class PromptAttention(nn.Module):
    def __init__(self, embed_dim, tag_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.tag_dim = tag_dim
        self.num_classes = num_classes

        self.query = nn.Linear(self.embed_dim, self.tag_dim)
        self.key = nn.Linear(self.embed_dim, self.tag_dim)
        self.value = nn.Linear(self.embed_dim, self.tag_dim)

        self.siameseNetwork = SiameseNetwork(self.embed_dim)

        self.linear1 = nn.Linear(self.tag_dim, 256)
        self.drop_1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, self.num_classes)
        self.drop_2 = nn.Dropout(p=0.5)
    
    def forward(self, X, Prompts_embeddings, Prompts_reference_essay_embeddings):
        '''
        X: [batch_size, dim]
        Prompts_emedding: [prompt_size, dim]
        '''
        
        # Implement attention mechanism
        # self.X_q = self.query(X)
        # self.X_k = self.key(X)
        # self.X_v = self.value(X)

        self.prompt_q = self.query(Prompts_embeddings) # Shape: (prompt_size, tag_dim)
        self.prompt_v = self.value(Prompts_embeddings) # Shape: (prompt_size, tag_dim)

        self.scores = torch.sigmoid(torch.matmul(X, self.prompt_q.transpose(0, 1))) # Shape: (batch_size, prompt_size)
        attn_weights = torch.softmax(self.scores, dim=-1) # batch, prompt_size => 每篇作文和对应prompt的重要性程度
        
        # output = torch.matmul(attn_weights, self.prompt_v)
        # output = self.drop_1(self.linear1(output)) # Shape: (batch_size, tag_dim)
        # output = self.drop_2(self.linear2(output))
        # output = torch.sigmoid(output)

        output = self.siameseNetwork(attn_weights, X, Prompts_reference_essay_embeddings, self.num_classes)

        return attn_weights, output # torch.Size([128, 2]) torch.Size([128, 8])
    
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.linear1.apply(clipper)
        self.linear2.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)




# # 定义模型参数
# embed_dim = 768
# tag_dim = 768

# # 实例化模型
# model = PromptAttention(embed_dim, tag_dim)

# # 生成随机输入数据
# X = torch.randn(1, embed_dim)  # 大小为 (1, 768)
# Prompts_emedding = torch.randn(3, embed_dim)  # 大小为 (3, 768)

# # 前向传播
# output, att = model(X, Prompts_emedding)

# # 打印输出
# print("Output shape:", output.shape)
# print(att)
