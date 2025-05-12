import jsonlines
import torch
import random

class TrainDataLoader(object):
    def __init__(self, processor, data_file, batch_size, Prompts_list, tag_dim, prompts_embeddings):
        self.processor = processor # 预训练模型
        self.batch_size = batch_size
        self.ptr = 0
        self.data = []
        self.data_file = data_file
        self.Prompts_list = Prompts_list
        self.tag_dim = tag_dim

        # self.processed_prompts = self.processor.process_list(self.Prompts_list) 
        self.processed_prompts = prompts_embeddings

        with jsonlines.open(data_file) as reader:
            self.data = [obj for obj in reader]
        
        random.shuffle(self.data)
    
    def next_batch(self):
        if self.is_end():
            return None
        
        batch_data = self.data[self.ptr:self.ptr + self.batch_size]
        self.ptr += self.batch_size
        
        
        batch_essay_texts_features = torch.stack([self.processor.process_singe_essay(item["Essay"]) for item in batch_data]).reshape(-1, self.tag_dim)
        batch_prompt_texts_features = torch.stack([self.processed_prompts[self.Prompts_list.index(item["Prompt"])]  for item in batch_data]).reshape(-1, self.tag_dim)

        batch_total_scores = torch.LongTensor([item["total_score"] for item in batch_data])
        batch_relevance_scores = torch.LongTensor([item["relevance_score"] for item in batch_data])
        batch_content_scores = torch.LongTensor([item["content_score"] for item in batch_data])
        batch_expression_scores = torch.LongTensor([item["expression_score"] for item in batch_data])
        
        return (
            batch_essay_texts_features, 
            batch_prompt_texts_features, 
            batch_total_scores, 
            batch_relevance_scores, 
            batch_content_scores, 
            batch_expression_scores
        )

    def is_end(self):
        return self.ptr >= len(self.data)

    def reset(self):
        self.ptr = 0
        random.shuffle(self.data)