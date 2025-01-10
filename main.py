import os, json
from retrieve import *
from utils import *
from dotenv import load_dotenv
from config import Config
from llmodel import LLModel
from typing import List, Dict
from summarizer import Summarizer
from FlagEmbedding import FlagModel, FlagReranker
import transformers
transformers.logging.set_verbosity_error()

def init():
    load_dotenv()
    # for llm api
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = Config()
    return config

def llmodels_init(config: Config):
    summarizer = Summarizer(config, stream=False,
                    model_name="qwen/qwen-2.5-72b-instruct", 
                    system_prompt_name="conversation_summary")

    # for main model
    legal_assistant_model = LLModel(config, summarizer=summarizer, stream=True,
                                    model_name="qwen/qwen-2.5-72b-instruct",
                                    system_prompt_name="legal_assistant")
    # for intent recognition
    intent_recog_model = LLModel(config, summarizer=summarizer, stream=True,
                      model_name="qwen/qwen-2.5-72b-instruct", 
                      system_prompt_name="intent_recognition")
    
    query_rewrite_model = LLModel(config, summarizer=summarizer, stream=False,
                    model_name="qwen/qwen-2.5-72b-instruct", 
                    system_prompt_name="query_rewrite")

    model_dict = {
        "legal_assistant_model": legal_assistant_model,
        "intent_recog_model": intent_recog_model,
        "query_rewrite_model": query_rewrite_model
    }
    return model_dict

def embedding_models_init(config: Config):
    embedding_model = FlagModel(
        model_name_or_path = config.flag_model_name,
        query_instruction_for_retrieval=config.query_instruction_for_retrieval,
        use_fp16=config.use_fp16
    )

    rerank_model = FlagReranker(config.rerank_model_name, use_fp16=config.use_fp16)

    return {"embedding_model": embedding_model, "rerank_model": rerank_model}

def retrieve_db(models: dict, rewritten_query: str, keywords: list[str], config: Config):
    print("Retrieving from database...")
    retrieved_content = get_context(models, rewritten_query, keywords, config)
    return retrieved_content

def generate_response(user_input: str, model: LLModel, print_response = True):
    model.add_user_message(user_input)
    response = model.get_response(print_response)
    model.add_assistant_message(response)
    # model.print_prompt_history()
    return response

def generate(model: LLModel, query: str, context: str):
    # concate the context and query
    messages = f"问题: {query}\n获取的内容: {context}"
    # print(messages)
    model.add_user_message(messages)
    response = model.get_response(print_response=True)
    model.remove_last_message()
    model.add_user_message(query)
    model.add_assistant_message(response)
    # model.print_prompt_history()
    return response

def bash_run(config: Config, models: Dict[str, LLModel], embedding_models: dict):
    legal_assistant_model = models["legal_assistant_model"]
    intent_recog_model = models["intent_recog_model"]
    query_rewrite_model = models["query_rewrite_model"]

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        
        # get the intent of the user's input
        intent_res = generate_response(user_input, intent_recog_model, print_response = False)
        # print("\n意图: ", intent_res)
        if "no" in intent_res.lower():
            # if user asks a question that is not related to law, remove the last user's message
            print("\nAssistant: 对不起，无法解答与法律无关的问题。")
            continue
        elif "yes" in intent_res.lower():
            # if user asks a question that is related to law, then rewrite the question
            rewritten_query = generate_response(user_input, query_rewrite_model, print_response = False)
            # print(rewritten_query)
            try:
                rewritten_query_json = json.loads(rewritten_query)
            except json.JSONDecodeError:
                print("\nAssistant: 对不起，无法解答这个问题。")
                continue
            rewritten_query = rewritten_query_json["rewritten_query"]
            keywords = rewritten_query_json["keywords"]

            # get the context from the database
            retrieved_context = retrieve_db(embedding_models, rewritten_query, keywords, config)

            # generate the response
            _ = generate(legal_assistant_model, rewritten_query, retrieved_context)
        else:
            print(f"\nAssistant: {intent_res}")

def main():
    config = init()
    models = llmodels_init(config)
    embedding_models = embedding_models_init(config)
    bash_run(config, models, embedding_models)

if __name__ == "__main__":
    main()