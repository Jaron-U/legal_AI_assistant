import os
from retrieve import *
from utils import *
from config import Config
from llmodel import LLModel

def retrieve(query: str):
    retrieved_docs = get_context(query)
    return retrieved_docs

def generate(qwen72b: LLModel, question: str, context: str):
    qwen72b.add_to_conversation_history("user", f"问题: {question}, 获取的内容: {context}")
    response = qwen72b.get_response()
    return response

def bash_run(qwen72b: LLModel):
    prompt = get_prompt("plain_prompt")
    qwen72b.add_to_conversation_history("system", prompt)

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        retrieved_context = retrieve(user_input)
        ai_response = generate(qwen72b, user_input, retrieved_context)
        full_response = qwen72b.print_response(ai_response)
        qwen72b.add_to_conversation_history("assistant", full_response)
        qwen72b.print_conversation_history()

if __name__ == "__main__":
    config = init()
    qwen72b = LLModel(config)
    bash_run(qwen72b)