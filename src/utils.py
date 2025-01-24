import json
from FlagEmbedding import FlagReranker

from src.llmodel import LLModel
from src.prompts import *

def _get_response_no_embed(model: LLModel, user_input:str) -> str:
    model.add_user_message(user_input)
    response = model.get_response(print_response=False)
    model.add_assistant_message(response)
    return response

def get_user_intent(model: LLModel, user_input:str) -> str:
    if model.sys_prompt_func is None:
        message = intent_recognizer_pt(user_input)
        model.temp_add_user_message(message)
        response = model.get_response(print_response=False)

        # model.print_messages()

        model.remove_last_message()
    else:
        _get_response_no_embed(model, user_input)
        # model.print_messages()
    return response

def _parse_json(json_str: str) -> Dict:
    try:
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        json_str = json_str[start_idx:end_idx]
        rewritten_query_json = json.loads(json_str)
        return rewritten_query_json
    except json.JSONDecodeError:
       return None

def get_rewritten_query(model: LLModel, user_input:str) -> Dict:
    if model.sys_prompt_func is None:
        message = query_rewriter_pt(user_input)
        model.temp_add_user_message(message)
        response = model.get_response(print_response=False)

        # model.print_messages()

        model.remove_last_message()
    else:
        _get_response_no_embed(model, user_input)
        # model.print_messages()
    rewritten_query_json = _parse_json(response)
    return rewritten_query_json

def db_content_check(model: LLModel, query_with_docs: str) -> str:
    if model.sys_prompt_func is None:
        message = db_content_checker_pt(query_with_docs)
        model.temp_add_user_message(message)
        response = model.get_response(print_response=False)

        # model.print_messages()

        model.remove_last_message()
    else:
        _get_response_no_embed(model, query_with_docs)
        # model.print_messages()
    return response

def rerank_docs(query, context_docs, reranker: FlagReranker, ranked_top_k: int=5):
    sentence_pairs = [[query, context] for context in context_docs]
    scores = reranker.compute_score(sentence_pairs)
    
    score_docs = [{"score": score, "content": content} for score, content in zip(scores, context_docs)]
    sorted_score_docs = sorted(score_docs, key=lambda x: x["score"], reverse=True)[:ranked_top_k]
    
    return [doc["content"] for doc in sorted_score_docs]

def get_final_response(model: LLModel, query: str, contexts: List[str], print_response = True) -> str:
    str_context = "\n\n".join(contexts)
    query_with_context = f"用户的问题：{query}\n\n获取的内容：\n{str_context}"

    if model.sys_prompt_func is None:
        message = legal_assistant_pt(query_with_context)
    else:
        message = query_with_context

    model.temp_add_user_message(message)
    response = model.get_response(print_response=print_response)
    model.remove_last_message()

    if model.sys_prompt_func is not None:
        model.add_user_message(query)
        model.add_assistant_message(response)

    # model.print_messages()
    return response