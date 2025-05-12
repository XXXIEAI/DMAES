import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

from sklearn.metrics import cohen_kappa_score
import torch.optim as optim

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        return output
    

class DA_MoE_AES(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(DA_MoE_AES, self).__init__()  # 必须先调用父类的 __init__ 方法
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        

        self.experts = MoELayer(self.num_experts, self.in_features, self.out_features)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(768, 9)
        # )

        # 相关性分类头：计算相似度 (x 和 prompt)
        self.classifier_relevance = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, 9)  # 相关性是单一的输出（相似度）
        )

        # 内容分类头：9分类
        self.classifier_content = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, 9)  # 内容分类（9 分类）
        )

        # 表达分类头：9分类，初始化为内容分类头的参数
        self.classifier_expression = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, 9)  # 表达分类（9 分类）
        )

        # 总分分类头：9分类，初始化为相关性、内容、表达头参数的平均值
        self.classifier_score = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.out_features, 9)  # 总分分类（9 分类）
        )
    

    def forward(self, prompt, x):
        feature = self.experts(x)

        relevance_combined =(prompt + x)/2
        # print(relevance_combined.shape)
        relevance_pred = self.classifier_relevance(relevance_combined)

        # 内容分类头：进行9分类
        content_pred = self.classifier_content(feature)

        # 表达分类头：进行9分类，参数由内容分类头初始化
        expression_pred = self.classifier_expression(feature)

        # 总分分类头：进行9分类，初始化为相关性、内容、表达的平均值
        score_pred = self.classifier_score(feature)

        return feature, relevance_pred, content_pred, expression_pred, score_pred
    
    def freeze_relevance(self):
        """Freeze the relevance classifier after training"""
        for param in self.classifier_relevance.parameters():
            param.requires_grad = False
    
    def freeze_content(self):
        """Freeze the content classifier after training"""
        for param in self.classifier_content.parameters():
            param.requires_grad = False
    
    def freeze_expression(self):
        """Freeze the expression classifier after training"""
        for param in self.classifier_expression.parameters():
            param.requires_grad = False
    
    
    def init_content_with_relevance(self):
        """Initialize content classifier with relevance classifier weights"""
        with torch.no_grad():
            # Copy relevance classifier weights to content classifier
            self.classifier_content[1].weight.copy_(self.classifier_relevance[1].weight)
            self.classifier_content[1].bias.copy_(self.classifier_relevance[1].bias)
    
    def init_expression_with_content_and_relevance(self):
        """Initialize expression classifier with average of content and relevance classifier weights"""
        with torch.no_grad():
            # 计算相关性分类头和内容分类头权重的平均值
            avg_weight = (self.classifier_content[1].weight + self.classifier_relevance[1].weight) / 2
            avg_bias = (self.classifier_content[1].bias + self.classifier_relevance[1].bias) / 2
            
            # 用计算得到的平均值初始化表达分类头
            self.classifier_expression[1].weight.copy_(avg_weight)
            self.classifier_expression[1].bias.copy_(avg_bias)
    
    def init_score_with_average_weights(self):
        """Initialize score classifier with average weights of relevance, content, and expression classifiers"""
        with torch.no_grad():
            # Average the weights and biases of the three heads
            relevance_weights = self.classifier_relevance[1].weight
            content_weights = self.classifier_content[1].weight
            expression_weights = self.classifier_expression[1].weight
            avg_weights = (relevance_weights + content_weights + expression_weights) / 3
            self.classifier_score[1].weight.copy_(avg_weights)

            relevance_bias = self.classifier_relevance[1].bias
            content_bias = self.classifier_content[1].bias
            expression_bias = self.classifier_expression[1].bias
            avg_bias = (relevance_bias + content_bias + expression_bias) / 3
            self.classifier_score[1].bias.copy_(avg_bias)


# num_experts = 3
# in_features = 768
# out_features = 768

# net = DA_MoE_AES(num_experts, in_features, out_features)

# x = torch.randn(32, 768)  
# prompt = torch.randn(32, 768)
# feature, relevance_score, content_pred, expression_pred, score_pred = net(prompt, x)
# print(feature.shape, relevance_score.shape, content_pred.shape, expression_pred.shape, score_pred.shape)


