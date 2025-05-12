import logging
import argparse
import pandas as pd
import time
from datetime import datetime
import pytz
import torch

beijing_tz = pytz.timezone("Asia/Shanghai") # 设置北京时间时区
utc_now = datetime.utcnow()
beijing_time = utc_now.astimezone(beijing_tz)
time_str = beijing_time.strftime("%Y-%m-%d-%H-%M")
log_file = f"./train_logs/train_log_"+str(time_str)+".txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*100)
print(f'running on {device} ...')
print("="*100)

parser = argparse.ArgumentParser(description="aes_code")  

parser.add_argument("--dataset", type=str, default="Elion", help="used dataset")
parser.add_argument("--num_epochs", type=int, default=100, help="epochs")
parser.add_argument("--batch_size", type=int, default=128, help="epochs")
parser.add_argument("--data_file", type=str, default='E:\CodeXXX\Prompt_Attention_250322\json_data\output_prompt_in_asap_2_8.jsonl', help="epochs")
parser.add_argument("--prompt_size", type=int, default=2, help="prompt_size")
parser.add_argument("--num_classes", type=int, default=3, help="prompt_size")
parser.add_argument("--reference_num_each_level", type=int, default=2, help="reference_sample_num")


df = pd.read_excel('/workspace/Prompt_Attention_250410/data/Elion/Elion_data_all.xlsx', sheet_name='Elion_target_data')

print(df.head())