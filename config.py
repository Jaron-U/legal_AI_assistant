import os
class Config:
    # for llm
    llm_api_url = "https://api.novita.ai/v3/openai"
    llm_api_key = os.getenv("OPENAI_API_KEY")

    # for FlagModel
    flag_model_name = 'BAAI/bge-large-zh-v1.5'
    query_instruction_for_retrieval = "为这个句子生成表示以用于检索相关文章："
    use_fp16 = True

    # for Elasticsearch
    db_url = "http://localhost:9200/legal_data/_search"

    # for retrieve
    db_k = 3

    web_k = 3