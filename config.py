import os
class Config:
    # for llm model(qwen/qwen-2.5-72b-instruct)
    qwen72b_model = "qwen/qwen-2.5-72b-instruct"
    llm_api_url = "https://api.novita.ai/v3/openai"
    llm_api_key = os.getenv("OPENAI_API_KEY")
    stream = True
    max_tokens = 32000
    temperature = 1
    top_p = 1
    presence_penalty = 0
    frequency_penalty = 0
    response_format = { "type": "text" }
    top_k = 50
    repetition_penalty = 1
    min_p = 0
    repetition_penalty = 1

    # for FlagModel
    flag_model_name = 'BAAI/bge-large-zh-v1.5'
    query_instruction_for_retrieval = "为这个句子生成表示以用于检索相关文章："
    use_fp16 = True

    # for Elasticsearch
    db_url = "http://localhost:9200/legal_data/_search"

    # for retrieve
    k = 3