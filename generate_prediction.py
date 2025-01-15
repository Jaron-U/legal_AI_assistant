import os, json
from legal_ai.retrieve import *
from legal_ai.utils import *
from dotenv import load_dotenv
from legal_ai.config import Config
from legal_ai.llmodel import LLModel
from typing import List, Dict
from legal_ai.summarizer import Summarizer
from FlagEmbedding import FlagModel, FlagReranker
import transformers
transformers.logging.set_verbosity_error()
from legal_ai.web_rerieve import web_retrieve
from main import *
from tqdm import tqdm

from rouge_chinese import Rouge
import jieba
from nltk.translate.gleu_score import corpus_gleu

def generate_prediction(config: Config, models: Dict[str, LLModel], embedding_models: dict):
    with open('evaluate/flzx.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    
    predications = {}
    
    for i, item in enumerate(tqdm(test_data)):
        query = item['question']

        response = only_response(query, config, models, embedding_models)
        print(response)
        answer = item['answer']

        curr = {
            str(i): {
                "origin_prompt": query,
                "prediction": response,
                "refr": answer
            }
        }

        predications.update(curr)

    with open('predictions.json', 'w', encoding='utf-8') as file:
        json.dump(predications, file, ensure_ascii=False, indent=4)

# from https://github.com/open-compass/LawBench/blob/main/evaluation/utils/function_utils.py#L32
def compute_rouge(hyps, refs):
    assert(len(hyps) == len(refs))
    hyps = [' '.join(jieba.cut(h)) for h in hyps]
    hyps = [h if h.strip() != "" else "无内容" for h in hyps]
    refs = [' '.join(jieba.cut(r)) for r in refs]
    return Rouge().get_scores(hyps, refs)
# from https://github.com/open-compass/LawBench/blob/main/evaluation/utils/function_utils.py#L32
def compute_flzx(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for key, entry in tqdm(data_dict.items()):
        question, prediction, answer = entry["origin_prompt"], entry["prediction"], entry["refr"]
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}


if __name__ == "__main__":
    config = init()
    models = llmodels_init(config)
    embedding_models = embedding_models_init(config)
    generate_prediction(config, models, embedding_models)

    # evaluate the predictions
    with open('predictions.json', 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    print("start to calculate the score...")
    print(compute_flzx(data_dict))
