import os
from config import Config
from dotenv import load_dotenv
from llmodel import LLModel

def init():
    load_dotenv()
    
    # for llm api
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = Config()
    return config

def query_intent(query: str, intent_model: LLModel):
    intent_model.add_to_conversation_only_history("user", query)
    raw_response = intent_model.get_response()
    response = intent_model.split_response(raw_response)
    return response