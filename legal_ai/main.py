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

def init():
    load_dotenv()
    # for llm api
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = Config()
    return config

def llmodels_init(config: Config):
    summarizer = Summarizer(config, stream=False,
                    model_name="qwen/qwen-2-7b-instruct", 
                    system_prompt_name="conversation_summary")

    # for main model
    legal_assistant_model = LLModel(config, summarizer=summarizer, stream=True,
                                    conversation_embedding_prompt=True,
                                    dialog_summary=False,
                                    max_round_dialog=1,
                                    model_name="qwen/qwen-2-7b-instruct",
                                    system_prompt_name="legal_assistant")
    # for intent recognition
    intent_recog_model = LLModel(config, summarizer=summarizer, stream=True,
                                conversation_embedding_prompt=True,
                                dialog_summary=False,
                                max_round_dialog=1,
                                model_name="qwen/qwen-2-7b-instruct", 
                                system_prompt_name="intent_recognition")
    
    query_rewrite_model = LLModel(config, summarizer=summarizer, stream=False,
                    conversation_embedding_prompt=True,
                    dialog_summary=False,
                    max_round_dialog=1,
                    model_name="qwen/qwen-2-7b-instruct", 
                    system_prompt_name="query_rewrite")

    db_content_check_model = LLModel(config, summarizer=summarizer, stream=True,
                                conversation_embedding_prompt=True,
                                dialog_summary=False,
                                model_name="qwen/qwen-2-7b-instruct", 
                                system_prompt_name="db_content_check")
    
    web_content_summarizer = Summarizer(config, stream=False,
                    model_name="qwen/qwen-2-7b-instruct", 
                    system_prompt_name="content_summary")

    model_dict = {
        "legal_assistant_model": legal_assistant_model,
        "intent_recog_model": intent_recog_model,
        "query_rewrite_model": query_rewrite_model,
        "web_content_summarizer": web_content_summarizer,
        "db_content_check_model": db_content_check_model
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
    # print("Retrieving from database...")
    retrieved_content = get_context(models, rewritten_query, keywords, config)

    # combine the ranked context into one string
    context = "\n\n".join(retrieved_content)
    return context

def retrieve_web(models: dict, embedding_models: dict, rewritten_query: str, 
                 db_content: str, keywords: list[str], config: Config):
    db_content_check_model: LLModel = models["db_content_check_model"]
    web_content_summarizer: Summarizer = models["web_content_summarizer"]
    # print("checking the content...")
    message = f"问题: {rewritten_query}\n\n获取的内容: {db_content}"
    # print(message)
    db_content_check_model.messages_embed(query=message, embed_history=False)
    response = db_content_check_model.get_response(print_response=False)
    # print("web_check_response: ", response)
    if "no" in response.lower():
        # print("Retrieving from the web...")
        web_content = web_retrieve(keywords, rewritten_query,
                                    reranker=embedding_models["rerank_model"], 
                                    max_results=config.web_k, ranked_k=config.web_ranked_k,
                                    content_summary_model=web_content_summarizer)
        return web_content
    else:
        return None

def generate_response(user_input: str, model: LLModel, print_response = True):
    model.messages_embed(user_input)
    response = model.get_response(print_response=print_response)
    model.add_user_message(user_input)
    model.add_assistant_message(response)
    # model.print_prompt_history()
    return response

def generate(model: LLModel, query: str, context: str, print_response = True):
    # concate the context and query
    message = f"问题: {query}\n\n获取的内容: {context}"
    # print(message)
    model.messages_embed(message)
    response = model.get_response(print_response=print_response)
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
        if "no" in intent_res.lower():
            # if user asks a question that is not related to law, remove the last user's message
            print("\nAssistant: 对不起，无法解答与法律无关的问题。")
            continue
        elif "yes" in intent_res.lower():
            # if user asks a question that is related to law, then rewrite the question
            rewritten_query = generate_response(user_input, query_rewrite_model, print_response = False)
            print(rewritten_query)
            try:
                start_idx = rewritten_query.find('{')
                end_idx = rewritten_query.rfind('}') + 1
                json_str = rewritten_query[start_idx:end_idx]
                rewritten_query_json = json.loads(json_str)
            except json.JSONDecodeError:
                print("\nAssistant: 对不起，无法解答这个问题。")
                continue
            rewritten_query = rewritten_query_json["rewritten"]
            keywords = rewritten_query_json["keywords"]

            # get the context from the database
            retrieved_context = retrieve_db(embedding_models, rewritten_query, keywords, config)

            # check the content from the database
            web_content = retrieve_web(models, embedding_models, rewritten_query, 
                                       retrieved_context, keywords, config)
            print("web_content: ", web_content)
            if web_content:
                retrieved_context += "\n\n" + web_content

            # generate the response
            _ = generate(legal_assistant_model, user_input, retrieved_context)
        else:
            print(f"\nAssistant: {intent_res}")




def retrieve_db_list(models: dict, rewritten_query: str, keywords: list[str], config: Config) -> List[str]:
    # print("Retrieving from database...")
    retrieved_content = get_context(models, rewritten_query, keywords, config)
    return retrieved_content
    
def only_response(user_input: str, config: Config, models: Dict[str, LLModel], embedding_models: dict):
    legal_assistant_model = models["legal_assistant_model"]
    intent_recog_model = models["intent_recog_model"]
    query_rewrite_model = models["query_rewrite_model"]

    retrieved_context = []
    
    # get the intent of the user's input
    intent_res = generate_response(user_input, intent_recog_model, print_response = False)
    print("intent_res: ", intent_res)
    if "no" in intent_res.lower():
        # if user asks a question that is not related to law, remove the last user's message
        return ("对不起，无法解答与法律无关的问题。"), retrieved_context
    elif "yes" in intent_res.lower():
        # if user asks a question that is related to law, then rewrite the question
        rewritten_query = generate_response(user_input, query_rewrite_model, print_response = False)
        print(rewritten_query)
        try:
            start_idx = rewritten_query.find('{')
            end_idx = rewritten_query.rfind('}') + 1
            json_str = rewritten_query[start_idx:end_idx]
            rewritten_query_json = json.loads(json_str)
        except json.JSONDecodeError:
            return ("对不起，无法解答这个问题。"), retrieved_context
        rewritten_query = rewritten_query_json["rewritten"]
        keywords = rewritten_query_json["keywords"]

        # get the context from the database
        retrieved_context = retrieve_db_list(embedding_models, rewritten_query, keywords, config)

        # check the content from the database
        web_content = retrieve_web(models, embedding_models, rewritten_query, 
                                    retrieved_context, keywords, config)
        # print("web_content: ", web_content)
        if web_content:
            retrieved_context.append(web_content)

        # generate the response
        response = generate(legal_assistant_model, user_input, retrieved_context, print_response=False)
        return response, retrieved_context
    else:
        return (f"Assistant: {intent_res}"), retrieved_context

def main():
    config = init()
    models = llmodels_init(config)
    embedding_models = embedding_models_init(config)
    bash_run(config, models, embedding_models)
    
    # user_input = "我们家老人去世后，他的女儿要求兄妹几个平分，但是我家一直赡养着老人，老人女儿很少在老人生病时进行照顾或者给钱"
    # print(only_response(user_input, config, models, embedding_models))

if __name__ == "__main__":
    main()