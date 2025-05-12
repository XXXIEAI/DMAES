import pandas as pd
import os
import glob
import json

'''
返回格式：
{
    "Prompt3": [
        {
            "Essay ID": 5978,
            "Essay": "XXXXXX ",
            "Content": 0.0,
            "Prompt Adherence": 0.0,
            "Language": 1.0,
            "Narrativity": 1.0,
            "domain1_score": 1.0,
            "final_score": 1.0
        },
        .....],
    "Prompt4": [
        {
            "Essay ID": 8863,
            "Essay": "XXXX",
            "Content": 0.0,
            "Prompt Adherence": 0.0,
            "Language": 0.0,
            "Narrativity": 0.0,
            "domain1_score": 0.0,
            "final_score": 0.0
        },
        .....],
'''


def get_csv_files(path):
    '''
    Prompt 1-2: 评分traits一致, Content/Organization/Word Choice/Sentence Fluency/Conventions
        Prompt 1: 1-6
        Prompt 2: 1-5
    Prompt 3-6: 评分traits一致, Content/Prompt Adherence/Language/Narrativity
        Prompt 3: 0-3
        Prompt 4: 0-3
        Prompt 5: 0-4
        Prompt 6: 0-4
    Prompt 7: Ideas_1/Organization_1/Style_1/Conventions_1
        Prompt 7: 1-12
    Prompt 8: IdeasAndContent_1/Organization_1/Voice_1/WordChoice_1/SentenceFluency_1/Conventions_1
        Prompt 8: 5-30
    '''
    return glob.glob(f"{path}/*.csv")


def return_essay_data(path):
    csv_files = get_csv_files(path)

    json_data = {}
    for csv_item in csv_files:
        '''.DS_Store'''
        
        title = ''.join(csv_item.split('.')[0].split('-')[:2]).split('/')[-1] # prompt名称

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

        json_data[title] = df.to_dict(orient="records")

    
    # with open("output.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=4)
    # with open("output.jsonl", "w", encoding="utf-8") as f:
    #     for key, value in json_data.items():
    #         json.dump({key: value}, f, ensure_ascii=False)
    #         f.write("\n")
    with open("/workspace/Prompt_Attention/json_data/output.jsonl", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    # with open("output.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


# return_essay_data('/workspace/Prompt_Attention/data/ASAP++_250317_3_6')


