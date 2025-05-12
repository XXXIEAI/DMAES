
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import jsonlines
from sklearn.metrics import cohen_kappa_score
from data_load import TrainDataLoader
from pretrain_mode_processor import PreTrainModelProcessor
from model import DA_MoE_AES
import logging
import time
from datetime import datetime
import pytz

beijing_tz = pytz.timezone("Asia/Shanghai") # 设置北京时间时区
utc_now = datetime.utcnow()
beijing_time = utc_now.astimezone(beijing_tz)
time_str = beijing_time.strftime("%Y-%m-%d-%H-%M")
log_file = f"./train_logs/train_log_"+str(time_str)+".txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

'''
Elion数据集 评分范围：
total_score: 0-8

relevance_score: 0-8
content_score: 0-8
expression_score: 0-8
'''


def train(epoch, task_item, model, dataloader, optimizer, loss_fun, device):
    model.train()  # 设置模型为训练模式
    data_loader.reset()
    running_loss = 0.0
    
    total_samples = 0
    batch_count = 0

    total_qwk_score = 0
    total_qwk_rel = 0
    total_qwk_con = 0
    total_qwk_exp = 0

    total_correct_scores = 0
    total_correct_rel = 0
    total_correct_con = 0
    total_correct_exp = 0
    
    if task_item == 'rel':
        while not data_loader.is_end():
            batch_count += 1
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = data_loader.next_batch()
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = essay_texts.to(device), prompt_texts.to(device), total_scores.to(device), relevance_scores.to(device), content_scores.to(device), expression_scores.to(device)
            
            optimizer.zero_grad()
            feature, relevance_pred, content_pred, expression_pred, score_pred = net.forward(essay_texts, prompt_texts)

            loss = loss_fun(relevance_pred, relevance_scores)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            rel_scores_pred = torch.argmax(relevance_pred, dim=1) 
                                
            qwk_rel = cohen_kappa_score(relevance_scores.cpu().numpy(), rel_scores_pred.cpu().numpy(), weights="quadratic")
            total_qwk_rel += qwk_rel
                
            total_samples += relevance_scores.size(0)  # 累计总样本数

        qwk_rel = total_qwk_rel / batch_count * 100

        loss = running_loss / batch_count
        logging.info(f"Epoch {epoch + 1}, loss: {loss:.4f}, QWK: {qwk_rel: .4f}")
        print(f'Epoch {epoch+1}, loss: {loss: .4f}, QWK_rel: {qwk_rel: .4f} ')
    
    elif task_item == 'content':

        while not data_loader.is_end():
            batch_count += 1
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = data_loader.next_batch()
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = essay_texts.to(device), prompt_texts.to(device), total_scores.to(device), relevance_scores.to(device), content_scores.to(device), expression_scores.to(device)
            
            optimizer.zero_grad()
            feature, relevance_pred, content_pred, expression_pred, score_pred = net.forward(essay_texts, prompt_texts)

            loss = loss_fun(content_pred, content_scores)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            con_scores_pred = torch.argmax(content_pred, dim=1) 
                                
            # qwk_con = cohen_kappa_score(content_pred.cpu().numpy(), con_scores_pred.cpu().numpy(), weights="quadratic")
            qwk_con = cohen_kappa_score(
                content_pred.detach().cpu().numpy(), 
                con_scores_pred.detach().cpu().numpy(), 
                weights="quadratic"
            )
            total_qwk_con += qwk_con
                
            total_samples += content_scores.size(0)  # 累计总样本数

        qwk_con = total_qwk_con / batch_count * 100

        loss = running_loss / batch_count
        logging.info(f"Epoch {epoch + 1}, loss: {loss:.4f}, QWK: {qwk_con: .4f}")
        print(f'Epoch {epoch+1}, loss: {loss: .4f}, QWK_con: {qwk_con: .4f}')
    
    elif task_item == 'expression':

        while not data_loader.is_end():
            batch_count += 1
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = data_loader.next_batch()
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = essay_texts.to(device), prompt_texts.to(device), total_scores.to(device), relevance_scores.to(device), content_scores.to(device), expression_scores.to(device)
            
            optimizer.zero_grad()
            feature, relevance_pred, content_pred, expression_pred, score_pred = net.forward(essay_texts, prompt_texts)

            loss = loss_fun(expression_pred, expression_scores)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            expression_pred = torch.argmax(expression_pred, dim=1) 
                                
            qwk_expression = cohen_kappa_score(expression_pred.detach().cpu().numpy(), 
                expression_scores_pred.detach().cpu().numpy(), weights="quadratic")
            total_qwk_expression += qwk_expression
                
            total_samples += expression_scores.size(0)  # 累计总样本数

        qwk_expression = total_qwk_expression / batch_count * 100

        loss = running_loss / batch_count
        logging.info(f"Epoch {epoch + 1}, loss: {loss:.4f}, QWK: {qwk_expression: .4f}")
        print(f'Epoch {epoch+1}, loss: {loss: .4f}, QWK_expression: {qwk_expression: .4f}')
    
    elif task_item == 'total_score':

        while not data_loader.is_end():
            batch_count += 1
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = data_loader.next_batch()
            essay_texts, prompt_texts, total_scores, relevance_scores, content_scores, expression_scores = essay_texts.to(device), prompt_texts.to(device), total_scores.to(device), relevance_scores.to(device), content_scores.to(device), expression_scores.to(device)
            
            optimizer.zero_grad()
            feature, relevance_pred, content_pred, expression_pred, score_pred = net.forward(essay_texts, prompt_texts)

            loss = loss_fun(score_pred, total_scores)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            score_pred = torch.argmax(score_pred, dim=1) 
                                
            qwk_score = cohen_kappa_score(score_pred.detach().cpu().numpy(), total_scores.detach().cpu().numpy(), weights="quadratic")
            total_qwk_score += qwk_score
                
            total_samples += total_scores.size(0)  # 累计总样本数

        qwk_score = total_qwk_score / batch_count * 100

        loss = running_loss / batch_count
        logging.info(f"Epoch {epoch + 1}, loss: {loss:.4f}, QWK: {qwk_score: .4f}")
        print(f'Epoch {epoch+1}, loss: {loss: .4f}, QWK_score: {qwk_score: .4f}')

        

def train_multitask(model, train_dataloader, num_epochs, learning_rate, device):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 定义损失函数
    loss_relevance = nn.CrossEntropyLoss()  # 相关性分类任务
    loss_content = nn.CrossEntropyLoss()  # 内容分类任务
    loss_expression = nn.CrossEntropyLoss()  # 表达分类任务
    loss_score = nn.CrossEntropyLoss()  # 总分分类任务
    
    model.to(device)  # 将模型移动到指定设备（GPU/CPU）

    task_types =['rel', 'content', 'expression', 'total_score']

    for task_item in task_types:
        print(f"Task: {task_item} Training ")
        print("="*100)

        if task_item == 'rel':
            # 训练相关性任务
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                train_loss = train(epoch, task_item, model, train_dataloader, optimizer, loss_relevance, device)
        
        elif task_item == 'content':
            # 冻结相关性分类头参数，并用相关性分类头的参数初始化内容分类头
            model.freeze_relevance()
            model.init_content_with_relevance()

            train_loss = train(epoch, task_item, model, train_dataloader, optimizer, loss_content, device)

        elif task_item == 'expression':
            # 冻结相关性、内容分类头参数，并用相关性、内容和表达分类头的参数平均值初始化表达分类头
            model.freeze_relevance()
            model.freeze_content()
            model.init_content_with_relevance()

            train_loss = train(epoch, task_item, model, train_dataloader, optimizer, loss_expression, device)

        elif task_item == 'total_score':
            model.freeze_relevance()
            model.freeze_content()
            model.freeze_expression()

            train_loss = train(epoch, task_item, model, train_dataloader, optimizer, loss_score, device)



Prompts_list = [
    "我的心爱之物",
    "“漫画”老师",
    "推荐一个好地方",
    "小小“动物园”",
    "二十年后的家乡",
    "我的动物朋友"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = PreTrainModelProcessor('/workspace/Prompt_Attention_250410/pretrain_models/bert-base-chinese', 
                                  '/workspace/Prompt_Attention_250410/pretrain_models/bert-base-chinese')

prompts_embeddings = processor.process_list(Prompts_list)
prompts_embeddings = torch.stack(prompts_embeddings).reshape(-1, 768).to(device)

data_loader = TrainDataLoader(
    processor, 
    '/workspace/Prompt_Attention_250410/json_data/Elion_target_data.jsonl', 
    batch_size = 128, 
    Prompts_list=Prompts_list, 
    tag_dim=768,
    prompts_embeddings=prompts_embeddings
    )



input_size = 768
output_size = 768
batch_size = 128
num_experts = 4

net = DA_MoE_AES(num_experts, input_size, output_size).to(device=device)
train_multitask(model=net, 
                 train_dataloader=data_loader, 
                 num_epochs=100, 
                 learning_rate=0.002, 
                 device=device)