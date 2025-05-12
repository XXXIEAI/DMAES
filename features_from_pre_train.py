from transformers import BertTokenizer, BertModel
import torch

class PreTrainModelProcessor:
    def __init__(self, model_path: str, tokenizer_path: str):
        # 加载预训练的 tokenizer 和 model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertModel.from_pretrained(model_path)

    def process_list(self, data_list):
        process_outputs = []

        # 处理每个 prompt
        for data in data_list:
            # 对文本进行编码
            encoded_input = self.tokenizer(data, 
                                            truncation=True,       # 截断超长文本
                                            max_length=512,
                                            return_tensors='pt')  # 'pt' 表示返回一个 PyTorch 张量（tensor）

            # 获取模型输出
            with torch.no_grad():  # 禁用梯度计算，节省内存
                output = self.model(**encoded_input)

            '''
            last_hidden_state: 保留了序列中每个位置的信息，包含了更丰富的上下文信息
            pooler_output: 将整个序列压缩为一个固定长度的向量
            '''
            # 获取 pooler_output
            # pooler_output = output.pooler_output
            last_output = output.last_hidden_state[:, 0, :]
            process_outputs.append(last_output)
        
        return process_outputs # type: list
    
    def process_singe_essay(self, essay):
        # encoded_input = self.tokenizer(essay, return_tensors='pt')
        encoded_input = self.tokenizer(essay, 
                                            truncation=True,       # 截断超长文本
                                            max_length=512,
                                            return_tensors='pt')
        with torch.no_grad():  # 禁用梯度计算，节省内存
            output = self.model(**encoded_input)

        last_output = output.last_hidden_state[:, 0, :]
        return last_output


# pre_train_model_path = '/workspace/Prompt_Attention/pretrain_models/bert-base-uncased'
# tokenizer_path = '/workspace/Prompt_Attention/pretrain_models/bert-base-uncased'

# processor = PreTrainModelProcessor(pre_train_model_path, tokenizer_path)
# prompt = "Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion."
# pooler_output = processor.process_singe_essay(prompt)
# print(pooler_output.shape)

