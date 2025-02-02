import json
from typing import List, Dict
import transformers
transformers.logging.set_verbosity_error()
from tqdm import tqdm

from rouge_chinese import Rouge
import jieba
from langchain_openai.chat_models.base import BaseChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from src.db_retrieve import *
from src.utils import *
from src.config import Config
from src.llmodel import LLModel
from src.main import *

def generate_prediction(config: Config, models: Dict[str, LLModel], embedding_models: dict):
    with open('evaluate/flzx.json', 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    
    predications = {}
    
    for i, item in enumerate(tqdm(test_data)):
        query = item['question']

        result_dict = for_evalute(query, models, embedding_models, config)
        answer = item['answer']

        curr = {
            str(i): {
                "origin_prompt": query,
                "query_rewriter": result_dict['rewritten_query'],
                "db_content_check": result_dict['db_content_check'],
                "contexts": result_dict['contexts'],
                "prediction": result_dict['response'],
                "refr": answer
            }
        }

        predications.update(curr)

    with open('evaluate_outputs/predictions_sft.json', 'w', encoding='utf-8') as file:
        json.dump(predications, file, ensure_ascii=False, indent=2)

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

def law_bench(config, models, embedding_models):
    # {'score': 0.17831063105163042}
    # qwen2-7b 0.159
    # sft: {'score': 0.1621620845886012}
    generate_prediction(config, models, embedding_models)

    # evaluate the predictions
    # {'score': 0.15733119902600465}
    with open('evaluate_outputs/predictions_sft.json', 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    print("start to calculate the score...")
    print(compute_flzx(data_dict))

def ragas_evaluate(config):
    evaluate_model = evaluate_model_init(config)

    with open('evaluate_outputs/predictions_sft.json', 'r', encoding='utf-8') as file:
        predictions = json.load(file)
    
    results = {
        'origin_prompt': [],
        'prediction': [],
        'contexts': [],
        'refr': []
    }
    for item in predictions.values():
        results['origin_prompt'].append(item['origin_prompt'])
        results['prediction'].append(item['prediction'])
        results['contexts'].append(item['contexts'])
        results['refr'].append(item['refr'])
    
    data = {
        "user_input": results['origin_prompt'],
        "response": results['prediction'],
        "retrieved_contexts": results['contexts'],
        "reference": results['refr']
    }

    dataset = Dataset.from_dict(data)
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluate_model,
    )

    df = results.to_pandas()
    averages = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean()
    averages.to_csv('averages_sft.csv', header=["average_value"])
    df.to_csv('ragas_evaluate_sft.csv')

def evaluate_model_init(config: Config):
    # llm = ChatOpenAI(
    #     model="gpt-4o",
    #     api_key=config.llm_api_key_evaluate,
    # )
    # return llm
    llm = BaseChatOpenAI(
        model=config.qwen7b, 
        openai_api_key=config.api_key, 
        openai_api_base=config.base_url,
        max_tokens=1024
    )
    # return llm

if __name__ == "__main__":
    config = Config()
    llms = llmodels_init(config)
    embedding_models = embedding_models_init(config)
    # law_bench(config, llms, embedding_models)

    ragas_evaluate(config)
