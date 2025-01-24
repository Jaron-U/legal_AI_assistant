from typing import List, Dict
from FlagEmbedding import FlagModel, FlagReranker
import transformers
transformers.logging.set_verbosity_error()

from src.config import Config
from src.llmodel import LLModel
from src.summarizer import Summarizer
from src.db_retrieve import retrieve_from_db
from src.web_retrieve import retrieve_from_web
from src.prompts import *
from src.utils import *

def llmodels_init(config: Config):
    summarizer = Summarizer(
        config.base_url, config.api_key,
        model_name=config.qwen7b, 
        prompt_func=dialog_summary_pt,
        max_tokens=512
    )

    intent_recognizer = LLModel(
        config.base_url, config.api_key,
        model_name=config.qwen7b,
        dialog_summary=False, max_round_dialog=0,
        obj_name="intent_recognizer",
        max_tokens=10
    )

    query_rewriter = LLModel(
        config.base_url, config.api_key,
        model_name=config.qwen7b, 
        dialog_summary=False,
        max_round_dialog=0, obj_name="query_rewriter"
    )

    db_content_checker = LLModel(
        config.base_url, config.api_key,
        model_name=config.qwen7b,
        dialog_summary=False, max_round_dialog=0,
        obj_name="db_content_checker",
        max_tokens=10
    )

    web_content_summarizer = Summarizer(
        config.base_url, config.api_key,
        model_name=config.qwen7b, 
        prompt_func=web_content_summary_pt,
        max_tokens=512
    )

    legal_assistant = LLModel(
        config.base_url, config.api_key,
        model_name=config.qwen7b, 
        # sys_prompt_func=legal_assistant_pt,
        dialog_summary=False, summarizer=summarizer, 
        max_round_dialog=3, obj_name="legal_assistant",
        min_round_dialog=1, stream = True, max_tokens = 512
    )

    return {
        "intent_recognizer": intent_recognizer,
        "query_rewriter": query_rewriter,
        "db_content_checker": db_content_checker,
        "web_content_summarizer": web_content_summarizer,
        "legal_assistant": legal_assistant,
    }

def embedding_models_init(config:Config):
    embedding_model = FlagModel(
        model_name_or_path=config.embedding_model_name,
        query_instruction_for_retrieval=config.embedding_model_instrcut,
        use_fp16=config.embedding_model_use_fp16,
    )

    rerank_model = FlagReranker(config.rerank_model_name, 
                                use_fp16=config.embedding_model_use_fp16)
    
    return {
        "embedding_model": embedding_model,
        "rerank_model": rerank_model,
    }

def bash_run(llms: Dict, embeding_models: Dict, config: Config):
    intent_recognizer = llms["intent_recognizer"]
    query_rewriter = llms["query_rewriter"]
    db_content_checker = llms["db_content_checker"]
    web_content_summarizer = llms["web_content_summarizer"]
    legal_assistant = llms["legal_assistant"]

    while True:
        user_input = input("\nuser: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        user_intent = get_user_intent(intent_recognizer, user_input)
        # print(f"intent: {user_intent}")
        if "yes" in user_intent.lower():
            rewritten_query_dict = get_rewritten_query(query_rewriter, user_input)
            print(f"\nrewritten_query_dict: {rewritten_query_dict}")
            if rewritten_query_dict is None:
                print(f"\nAssistant: 对不起，无法解答这个问题。")
                continue

            # get the context from the database
            print("\nRetrieving from database...")
            context_docs = retrieve_from_db(rewritten_query_dict, embeding_models, config)
            if context_docs is None:
                print(f"\nAssistant: 对不起，数据库查询失败。")
                continue
            
            # check if the query can be answered from the context
            db_content_check_res = db_content_check(db_content_checker, context_docs)
            # print(f"\ncontent check: {db_content_check_res}")
            web_context_summary = retrieve_from_web(db_content_check_res, web_content_summarizer, 
                                                    rewritten_query_dict, embeding_models['rerank_model'], 
                                                    config)
            if web_context_summary is not None:
                context_docs.append(web_context_summary)

            response = get_final_response(legal_assistant, user_input, context_docs)
        else:
            print("\nAssistant: 对不起，无法解答与法律无关的问题。")

def for_evalute(user_input: str, llms: Dict, embeding_models: Dict, config: Config) -> Dict:
    intent_recognizer = llms["intent_recognizer"]
    query_rewriter = llms["query_rewriter"]
    db_content_checker = llms["db_content_checker"]
    web_content_summarizer = llms["web_content_summarizer"]
    legal_assistant = llms["legal_assistant"]

    result = {
        "contexts": [],
        "rewritten_query": "",
        "db_content_check": "",
        "response": ""
    }

    user_intent = get_user_intent(intent_recognizer, user_input)
    if "yes" in user_intent.lower():
        rewritten_query_dict = get_rewritten_query(query_rewriter, user_input)
        print(f"\nrewritten_query_dict: {rewritten_query_dict}")
        if rewritten_query_dict is None:
            result["response"] = "对不起，无法解答这个问题。"
            return result
    
        result['rewritten_query'] = rewritten_query_dict

        context_docs = retrieve_from_db(rewritten_query_dict, embeding_models, config)
        if context_docs is None:
            result["response"] = "对不起，数据库查询失败。"
            return result
        
        # check if the query can be answered from the context
        db_content_check_res = db_content_check(db_content_checker, context_docs)
        result["db_content_check"] = db_content_check_res
        # print(f"\ncontent check: {db_content_check_res}")
        web_context_summary = retrieve_from_web(db_content_check_res, web_content_summarizer, 
                                                rewritten_query_dict, embeding_models['rerank_model'], 
                                                config)
        if web_context_summary is not None:
            context_docs.append(web_context_summary)
        result["contexts"] = context_docs
        response = get_final_response(legal_assistant, user_input, context_docs, print_response=False)
        result["response"] = response

    else:
        print("\nAssistant: 对不起，无法解答与法律无关的问题。")
        result["response"] = "对不起，无法解答与法律无关的问题。"
    
    return result

if __name__ == "__main__":
    config = Config()
    llms = llmodels_init(config)
    embedding_models = embedding_models_init(config)
    bash_run(llms, embedding_models, config)
