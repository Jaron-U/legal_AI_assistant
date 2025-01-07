import requests
from FlagEmbedding import FlagModel
import transformers
transformers.logging.set_verbosity_error()
from config import Config

config = Config()

def get_embedding_model(model_name=config.flag_model_name):
    model = FlagModel(
        model_name,
        query_instruction_for_retrieval=config.query_instruction_for_retrieval,
        use_fp16=config.use_fp16
    )
    return model

def get_query_vector(query):
    model = get_embedding_model()
    vector = model.encode(query)
    if isinstance(vector, list) is False:
        vector = vector.tolist() 
    return  vector

def search_query(query_vector):
    query = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match_all": {}}
                        ]
                    }
                },
                "script": {
                    "source": """
                        double paraScore = cosineSimilarity(params.queryVector, doc['para_vector']) + 1.0;
                        double titleScore = cosineSimilarity(params.queryVector, doc['title_vector']) + 1.0;
                        return paraScore + titleScore;
                    """,
                    "params": {
                        "queryVector": query_vector
                    }
                }
            }
        }
    }
    return query

def get_context(question, k=config.k, db_url=config.db_url):
    headers = {"Content-Type": "application/json"}
    query_vector = get_query_vector(question)
    query = search_query(query_vector)
    response = requests.post(db_url, json=query, headers=headers)
    if response.status_code == 200:
        hits = response.json().get('hits', {}).get('hits', [])[:k]
        context = "\n\n".join(
            f"{hit.get('_source', {}).get('title', '无标题')}\n{hit.get('_source', {}).get('para', '无内容')}"
            for hit in hits
        )
    else:
        context = "查询失败"
    return context
