import requests
from FlagEmbedding import FlagModel, FlagReranker
import transformers
transformers.logging.set_verbosity_error()
from config import Config

def get_query_vectors(model: FlagModel, keywords):
    vectors = {keyword: model.encode(keyword).tolist() for keyword in keywords}
    return vectors

def search_query(query_vectors: dict[str, list[float]]):
    query_conditions = []

    for keyword, vector in query_vectors.items():
        query_conditions.append({
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": """
                        double paraScore = cosineSimilarity(params.queryVector, doc['para_vector']) + 1.0;
                        double titleScore = cosineSimilarity(params.queryVector, doc['title_vector']) + 1.0;
                        return paraScore + titleScore;
                    """,
                    "params": {
                        "queryVector": vector
                    }
                }
            }
        })
    
    # Combine all conditions using `should`
    query = {
        "query": {
            "bool": {
                "should": query_conditions
            }
        }
    }
    
    return query

def _get_context_from_db(query_vectors, k = 10, db_url="http://localhost:9200/legal_data/_search"):
    headers = {"Content-Type": "application/json"}
    post_query = search_query(query_vectors)
    response = requests.post(db_url, json=post_query, headers=headers)
    if response.status_code == 200:
        hits = response.json().get('hits', {}).get('hits', [])[:k]
        context = [
            f"{hit.get('_source', {}).get('title', '无标题')} {hit.get('_source', {}).get('para', '无内容')}"
            for hit in hits
        ]
    else:
        context = "查询失败"
    return context

def rerank_documents(rewritten_query: str, context_hits: list[str], top_n, reranker: FlagReranker):
    # combine the context hits and query.
    sentence_pairs = [[rewritten_query, context] for context in context_hits]
    # calculate the reranking score
    scores = reranker.compute_score(sentence_pairs)

    score_document = [{"score": score, "content": content} for score, content in zip(scores, context_hits)]
    sorted_score_document = sorted(score_document, key=lambda x: x["score"], reverse=True)[:top_n]

    return [doc["content"] for doc in sorted_score_document]

def get_context(models: dict, rewritten_query:str, keywords: list[str], config: Config):
    embedding_model = models["embedding_model"]
    reranker = models["rerank_model"]
    keywords_vectors = get_query_vectors(embedding_model, keywords)
    context_hits = _get_context_from_db(keywords_vectors, config.db_k, config.db_url)
    reranked_context = rerank_documents(rewritten_query, context_hits, config.ranked_k, reranker)

    # combine the ranked context into one string
    context = "\n\n".join(reranked_context)

    return context

if __name__ == "__main__":
    def embedding_models_init(config: Config):
        embedding_model = FlagModel(
            model_name_or_path = config.flag_model_name,
            query_instruction_for_retrieval=config.query_instruction_for_retrieval,
            use_fp16=config.use_fp16
        )

        rerank_model = FlagReranker(config.rerank_model_name, use_fp16=config.use_fp16)

        return {"embedding_model": embedding_model, "rerank_model": rerank_model}
    keywords = ["民法商法", "农民专业合作社法", "第三十三条"]
    rewritten_query = "用户意图获取民法商法农民专业合作社法第三十三条的具体条款内容"
    config = Config()
    models = embedding_models_init(config)
    context = get_context(models, rewritten_query, keywords, config)
    print(context)

