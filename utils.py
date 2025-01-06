import os
from dotenv import load_dotenv

def set_api_keys():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def get_llm(model_name = "qwen/qwen-2.5-72b-instruct"):
    openai_api_key = os.getenv("OPENAI_API_KEY")