import pandas as pd
import os
from tqdm import tqdm
import json

def separate_items_csv(csv_file_path:str, single_output, multiple_output, chunksize=10):
    # there some questions in this dataset has multiple answers. 
    # this function is separate these this questions to different files

    current_group = []
    current_title = None

    single_count = 0
    multiple_count = 0
    multiple_group_count = 0

    def process_group(group):
        nonlocal single_count, multiple_count, multiple_group_count
        if not group:
            return 

        if len(group) == 1:
            pd.DataFrame(group).to_csv(single_output, mode='a', header=False, index=False)
            single_count += 1
        else:
            pd.DataFrame(group).to_csv(multiple_output, mode='a', header=False, index=False)
            multiple_count += len(group)
            multiple_group_count += 1
    
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize):
        for _, row in tqdm(chunk.iterrows()):
            if row['title'] != current_title:
                process_group(current_group)
                current_group = [row.to_dict()]
                current_title = row['title']
            else:
                current_group.append(row.to_dict())
        break
    
    process_group(current_group)

    print(f"single_count: {single_count}")
    print(f"multiple_count: {multiple_count}")
    print(f"multiple_group_count: {multiple_group_count}")

def format_csv2sharegpt(single_data_path, output_path):
    json_data = []
    for chunk in pd.read_csv(single_data_path, chunksize=15000):
        for idx in tqdm(range(len(chunk)), desc="Processing rows"):
            row = chunk.iloc[idx]
            if pd.notna(row.iloc[1]) and str(row.iloc[1]).strip():
                human = str(row.iloc[0]) if len(str(row.iloc[0])) > len(str(row.iloc[1])) else str(row.iloc[1])
            else:
                human = str(row.iloc[0])
            
            conversation = {
                "messages": [
                    {"role": "user", "content": human.strip()},
                    {"role": "assistant", "content": str(row.iloc[2]).strip()}
                ]
            }
            
            if human.strip() and pd.notna(row.iloc[2]):
                json_data.append(conversation)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print("done")

def process_csv(root_path):
    csv_file_path = "lawzhidao_filter.csv"
    csv_file_path = os.path.join(root_path, csv_file_path)
    single_data_path = os.path.join(root_path, "single_output.csv")
    # multiple_data_path = os.path.join(root_path, "multiple_output.csv")
    # separate_items_csv(csv_file_path, single_data_path, multiple_data_path, chunksize=50000)
    format_csv2sharegpt(single_data_path, "legal_sft_sharegpt_1.json")

def process_json(root_path):
    json_file_path = "qa_corpus.json"
    qs_json_path = os.path.join(root_path, json_file_path)
    output_path = os.path.join(root_path, "finetune", "legal_sft_sharegpt_2.json")

    json_data = []
    with open(qs_json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                if line.strip():
                    json_obj = json.loads(line)
                    user = json_obj['question']
                    answer = " ".join(json_obj['answers'])
                    messages = {
                        "messages": [
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    json_data.append(messages)
            except json.JSONDecodeError as e:
                print(f"error: {e}")
                continue
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print("done")

if __name__ == "__main__":
    root_path = "/home/jianglongyu/Documents/datasets/legal_dataset"

    # process_csv(root_path)
    process_json(root_path)