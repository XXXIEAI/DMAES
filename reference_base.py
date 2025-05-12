'''
从数据中获取一定的参考样本来计算相似度

由于做的是cross-promopt场景，所以获取不同prompt的参考样例的不同分数所对应的作文，然后取平均值

需要的变量：
1. 总的数据集
2. 每个level需要的reference的样本数量

返回格式：每个level所需要reference的样本数量的平均向量
'''


import json
import torch
import random
from collections import defaultdict
import pdb

class ReferenceBase(object):
    def __init__(self, args, procesor, device):
        self.json_data = args.data_file 
        self.reference_num_each_level = args.reference_num_each_level
        self.processor  = procesor
        self.device = device
        self.prompt_size = args.prompt_size

        with open(self.json_data, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        random.shuffle(self.data)

    def get_samples_by_prompt_and_score(self):
        grouped_data = defaultdict(lambda: defaultdict(list))
        
        # 按 Prompt_id 和 final_score 分组
        for item in self.data:
            grouped_data[item['Prompt_id']][item['final_score']].append(item)

        # 确保 Prompt_id 从 0 开始有序存储
        sorted_prompt_ids = sorted(grouped_data.keys())  # 按 Prompt_id 排序 [3, 4]

        processed_essays = []
        final_scores = []
        prompt_ids = []

        for prompt_id in sorted_prompt_ids:  # 按 Prompt_id 进行遍历
            score_dict = grouped_data[prompt_id] # dict_keys([1.0, 2.0, 3.0, 0.0])
            for score in sorted(score_dict.keys()):  # 进一步按 final_score 排序
                items = score_dict[score]
                selected_samples = random.sample(items, min(len(items), self.reference_num_each_level))

                if self.reference_num_each_level == 1:
                    for sample in selected_samples:
                        processed_essays.append(self.processor.process_singe_essay(sample["Essay"]).to(self.device))  # [1, 768]
                        prompt_ids.append(sample["Prompt_id"])
                        final_scores.append(sample["final_score"])
                        

                elif self.reference_num_each_level > 1:
                    for sample in selected_samples:
                        processed_essays.append(self.processor.process_singe_essay(sample["Essay"]).to(self.device))  # [1, 768]
                        final_scores.append(sample["final_score"])
                        prompt_ids.append(sample["Prompt_id"])
                    


        # 将列表转换为 PyTorch 张量并移动到 device
        essay_tensor = torch.cat(processed_essays, dim=0).to(self.device)  # [prompt_size * reference_num_each_level, 768]
        essay_tensor = essay_tensor.view(-1, self.reference_num_each_level , essay_tensor.shape[-1])  # [N/2, 2, 768]
        essay_mean = essay_tensor.mean(dim=1)  # [prompt_size, 768]
        
        final_score_tensor = torch.tensor(final_scores, dtype=torch.float32, device=self.device).unsqueeze(-1)  # [batch_size, 1]
        selected_indices = torch.arange(0, final_score_tensor.shape[0], step=self.reference_num_each_level, device=final_score_tensor.device)
        final_score_tensor = final_score_tensor[selected_indices]

        prompt_id_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(-1)  # [batch_size, 1]
        prompt_id_tensor = prompt_id_tensor[selected_indices]

        # 拼接成一个最终张量
        # print(essay_mean.shape, final_score_tensor, prompt_id_tensor)
        combined_tensor = torch.cat([essay_mean, final_score_tensor, prompt_id_tensor], dim=1)  # [batch_size, 770]

        return combined_tensor


# references = ReferenceBase("/workspace/Prompt_Attention/json_data/output_prompt_in_asap_3_4.jsonl", 4)
# samples = references.get_samples_by_prompt_and_score()
# for item in samples:
#     print(item["final_score"], item["Prompt_id"])

