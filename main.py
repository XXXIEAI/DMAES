import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pandas as pd
import pdb
import json
import logging
import time
from sklearn.metrics import cohen_kappa_score

from features_from_pre_train import PreTrainModelProcessor
from data_load.dataload import TrainDataLoader
from prompt_att_model import PromptAttention
from reference_base import ReferenceBase


timestamp = time.time()
time_tuple = time.localtime(timestamp)
time_str = time.strftime("%Y-%m-%d-%H-%M", time_tuple)
timezone_name = time.tzname
print("time zone: ", timezone_name)
log_file = f"./train_log_"+str(time_str)+".txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*100)
print(f'running on {device} ...')
print("="*100)


# 参数
parser = argparse.ArgumentParser(description="aes_code")  

parser.add_argument("--dataset", type=str, default="ASAP++", help="used dataset")
parser.add_argument("--num_epochs", type=int, default=100, help="epochs")
parser.add_argument("--batch_size", type=int, default=128, help="epochs")
parser.add_argument("--data_file", type=str, default='E:\CodeXXX\Prompt_Attention_250322\json_data\output_prompt_in_asap_2_8.jsonl', help="epochs")
parser.add_argument("--prompt_size", type=int, default=2, help="prompt_size")
parser.add_argument("--num_classes", type=int, default=3, help="prompt_size")
parser.add_argument("--reference_num_each_level", type=int, default=2, help="reference_sample_num")

# 预训练模型路径
parser.add_argument("--pre_train_model_path", default="./pretrain_models/bert-base-uncased", type=str)
parser.add_argument("--tokenizer_path", default="./pretrain_models/bert-base-uncased", type=str)
parser.add_argument("--embed_dim", type=int, default=768)
parser.add_argument("--tag_dim", type=int, default=768)

args = parser.parse_args()

logging.info(f"Parsed arguments: {vars(args)}")

assert args.dataset in ["ASAP++", "Elion"]

# pre_train_model_path = './pretrain_models/bert-base-uncased'
# tokenizer_path = './pretrain_models/bert-base-uncased'

processor = PreTrainModelProcessor(args.pre_train_model_path, args.tokenizer_path)

# 每个prompt的特征向量
# pooler_outputs = processor.process_prompts(Prompts_list)

if args.dataset == 'ASAP++':
    Prompts_list = [
        "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.",
        "Write a persuasive essay to a newspaper reflecting your vies on censorship in libraries. Do you believe that certain materials, such as books, music, movies, magazines, etc., should be removed from the shelves if they are found offensive? Support your position with convincing arguments from your own experience, observations, and/or reading.",
        "Write a response that explains how the features of the setting affect the cyclist. In your response, include examples from the essay that support your conclusion.",
        "Read the last paragraph of the story: When they come back, Saeng vowed silently to herself, in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again. Write a response that explains why the author concludes the story with this paragraph. In your response, include details and examples from the story that support your ideas.",
        "\"Narciso Rodriguez\" by Narciso Rodriguez, from Home: The Blueprints of Our Lives. Copyright © 2006 by John Edwards. Describe the mood created by the author in the memoir. Support your answer with relevant and specific information from the memoir.",
        "\"The Mooring Mast\" by Marcia Amidon Lüsted, from The Empire State Building. Copyright © 2004 by Gale, a part of Cengage Learning, Inc.Based on the excerpt, describe the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there. Support your answer with relevant and specific information from the excerpt.",
        "Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining.",
        "We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part."
    ]

    '''
    Prompt 1: 1-6
    Prompt 7: 1-12 (1-6)
    
    Prompt 2: 1-3
    Prompt 8: 5-30 (1-3)

    Prompt 3: 0-3
    Prompt 4: 0-3

    Prompt 5: 0-4
    Prompt 6: 0-4
    '''

    # 对prompt题目信息进行embedding
    prompts_embeddings = processor.process_list(Prompts_list)
    prompts_embeddings = torch.stack(prompts_embeddings).reshape(-1, 768).to(device)
 
    data_loader = TrainDataLoader(processor, args.data_file, args.batch_size, Prompts_list, args.tag_dim)

    # 参考样本
    references_object = ReferenceBase(args, processor, device)
    references_info = references_object.get_samples_by_prompt_and_score()
    

    references_info_embeddings, references_info_lables = references_info[:, :768].to(device), references_info[:, 768:].to(device)

    net = PromptAttention(args.embed_dim, args.tag_dim, args.num_classes).to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    
    print('training model...')

    # 不需要对 prompt_scores 使用 argmax(), softmax（cross_entropy 内部已经包含 softmax 计算）
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        data_loader.reset()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_count = 0
        total_qwk_score = 0
        while not data_loader.is_end():
            batch_count += 1
            essasy_embeddings, labels, prompt_scores = data_loader.next_batch()
            essasy_embeddings, labels, prompt_scores = essasy_embeddings.to(device), labels.to(device), prompt_scores.to(device)
            
            optimizer.zero_grad()
            attn_weights, outputs = net.forward(essasy_embeddings, prompts_embeddings, references_info_embeddings)

            max_indices = torch.argmax(attn_weights, dim=1)
            output_slice = torch.stack([outputs[i, args.num_classes*max_indices[i]:args.num_classes*max_indices[i]+args.num_classes] for i in range(max_indices.size(0))])

            loss_mask = loss_function(attn_weights, prompt_scores)
            loss_pred = loss_function(output_slice, labels)

            print(labels.min(), labels.max())  # 检查最小和最大类别索引


            alpha, beta = 0.8, 1.6          
            total_loss = alpha * loss_mask + alpha * loss_pred
            # print(loss_mask, loss_preed)
            # pdb.set_trace()

            total_loss.backward()
            optimizer.step()
            # net.apply_clipper()

            running_loss += total_loss.item()

            mask_pred = torch.argmax(attn_weights, dim=1) 
            label_preds = torch.argmax(output_slice, dim=1)  

                        
            qwk_score = cohen_kappa_score(labels.cpu().numpy(), label_preds.cpu().numpy(), weights="quadratic")
            total_qwk_score += qwk_score

            correct = (label_preds == labels).sum().item()  # 计算正确预测数
            total_correct += correct
            total_samples += labels.size(0)  # 累计总样本数

        epoch_acc = total_correct / total_samples * 100
        qwk = total_qwk_score / batch_count * 100
        loss = running_loss / batch_count
        logging.info(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {epoch_acc: .4f}, QWK: {qwk: .4f}")
        print(f'Epoch [%d] | loss: %.2f%%, Accuracy: %.2f%%' % (epoch + 1, loss, epoch_acc))


    
    '''
        Prompt 1: 1-6
        Prompt 2: 1-3
        Prompt 3: 0-3
        Prompt 4: 0-3
        Prompt 5: 0-4
        Prompt 6: 0-4
        Prompt 7: 1-12 (1-6), 在ASAP++_250318_1_7文件中, 将Prompt 7的ROUND(final_score/2), 划分为1-6
        Prompt 8: 5-30 (1-3), 在ASAP++_250318_2_8文件中, 将Prompt 8的=IF(A1<=13, 1, IF(A1<=21, 2, 3)), 划分为1-3
    '''

    '''
    Prompt 1: 1-6
    Prompt 7: 1-12 (1-6)
    
    Prompt 2: 1-3
    Prompt 8: 5-30 (1-3)

    Prompt 3: 0-3
    Prompt 4: 0-3

    Prompt 5: 0-4
    Prompt 6: 0-4
    '''
    
    





'''
    P_i  : Prompt
    E_ij : Essay
    S_ij : Score

    在cross-prompt场景下，需要对不用Prompt泛化：
        1. 需要检查不同prompt和对应essay的对应关系 => prompt和essay之间的权重关系
        2. 
    
    Q, K, V
    P_i * K, P_i * 
'''



