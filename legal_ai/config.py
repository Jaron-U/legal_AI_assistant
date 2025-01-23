import os
class Config:
    # for llm
    llm_api_url = "https://api.novita.ai/v3/openai"
    # llm_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key = os.getenv("OPENAI_API_KEY_C")

    local_api_url = "https://localhost:8000/v1"
    local_api_key = "none"

    llm_api_key_evaluate = os.getenv("OPENAI_API_KEY")

    # for FlagModel
    flag_model_name = 'BAAI/bge-large-zh-v1.5'
    query_instruction_for_retrieval = "为这个句子生成表示以用于检索相关文章："
    use_fp16 = True
    # flagReranker
    rerank_model_name = 'BAAI/bge-reranker-large'
    ranked_k = 5

    # for Elasticsearch
    db_url = "http://localhost:9200/legal_data/_search"

    # for retrieve for db
    db_k = 10

    # for retrieve for web
    web_k = 15
    web_ranked_k = 3

    conversation_buffer_keep_k = 3