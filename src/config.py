import os
from dotenv import load_dotenv
class Config:
    load_dotenv()

    # novita
    base_url = "https://api.novita.ai/v3/openai"
    api_key = os.getenv("OPENAI_API_KEY_C")
    qwen7b = "qwen/qwen-2-7b-instruct"

    # openai
    api_key_evaluate = os.getenv("OPENAI_API_KEY")

    # local
    local_base_url = "http://localhost:8000/v1"
    local_api_key = "none"
    local_qwen7b = "test"
    
    # for FlagModel
    embedding_model_name = 'BAAI/bge-large-zh-v1.5'
    embedding_model_instrcut = "为这个句子生成表示以用于检索相关文章："
    embedding_model_use_fp16 = True
    # flagReranker
    rerank_model_name = 'BAAI/bge-reranker-large'
    
    # for Elasticsearch
    db_url = "http://localhost:9200/legal_data/_search"

    # for retrieve for db
    retrieved_db_top_k = 15
    ranked_db_top_k = 5
    
    # for retrieve for web
    retrieved_web_top_k = 20
    ranked_web_top_k = 8