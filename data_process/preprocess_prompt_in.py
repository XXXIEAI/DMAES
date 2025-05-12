import pandas as pd
import os
import glob
import json

'''
返回格式：
[

    {
        "Essay ID": 5978,
        "Essay": "XXXXXX ",
        "Content": 0.0,
        "Prompt Adherence": 0.0,
        "Language": 1.0,
        "Narrativity": 1.0,
        "domain1_score": 1.0,
        "final_score": 1.0,
        "Prompt": "Prompt3"
    },
    .....
]
'''


def get_csv_files(path):
    return glob.glob(f"{path}/*.csv")


def return_essay_data(path):
    csv_files = get_csv_files(path)
    print(csv_files)

    json_data = []
    for csv_item in csv_files:
        
        # title = ''.join(csv_item.split('.')[0].split('-')[:2]).split('/')[-1] # prompt名称
        title = ''.join(csv_item.split('.')[0].split('/')[-1] )
        print(title)
        id = int(csv_item.split('.')[0].split('/')[-1].split('-')[-1])

        df = pd.read_csv(csv_item, encoding='ISO-8859-1', dtype={
            'EssayID': int, 
            'Essay': str,
            "Content": float,
            "Prompt Adherence":float,
            "Language":float,
            "Narrativity":float,
            "domain1_score":float,
            "final_score":float})
        df = df.dropna()

        # 为每条记录添加 Prompt 字段
        df['Prompt'] = title
        df['Prompt_id'] = id

        json_data.extend(df.to_dict(orient='records'))


    with open("/mnt/workspace/program/Prompt_Attention_1/json_data/output_prompt_in_asap_5_6.jsonl", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)



# return_essay_data('/mnt/workspace/program/Prompt_Attention_1/data/ASAP++_250318_5_6')


