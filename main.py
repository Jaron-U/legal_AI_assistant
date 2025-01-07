import os
from retrieve import *
from utils import *
from config import Config
from llmodel import LLModel

def retrieve(query: str):
    retrieved_docs = get_context(query)
    return retrieved_docs

def generate(qwen72b: LLModel, question: str, context: str):
    qwen72b.add_to_conversation_only_history("user", f"问题: {question}, 获取的内容: {context}")
    response = qwen72b.get_response()
    return response

def bash_run(qwen72b: LLModel, intent_model: LLModel):
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        intent_res = query_intent(user_input, intent_model)

        if "no" in intent_res.lower():
            intent_model.remove_last_message()
            print("\nAssistant: 对不起，无法解答与法律无关的问题。")
            continue
        elif "yes" in intent_res.lower():
            retrieved_context = retrieve(user_input)
            ai_response = generate(qwen72b, user_input, retrieved_context)
            full_response = qwen72b.print_response(ai_response)
            qwen72b.add_to_conversation_only_history("assistant", full_response)
        else:
            print("\nAssistant: ", intent_res)

if __name__ == "__main__":
    config = init()
    # for main model
    qwen72b = LLModel(config)

    # for intent recognition
    qwen7b = LLModel(config, stream=False,
                      model_name="qwen/qwen-2-7b-instruct", 
                      system_prompt_name="intent_recognition")

    bash_run(qwen72b, qwen7b)