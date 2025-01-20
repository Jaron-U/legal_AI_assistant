import requests
from FlagEmbedding import FlagModel, FlagReranker
import transformers
transformers.logging.set_verbosity_error()
from config import Config

####################
# def get_query_vectors(model: FlagModel, rewrite_query):
#     vectors = model.encode(rewrite_query).tolist()
#     return vectors

# def search_query(query_vectors: list[float]=None, keywords: list[str]=None):
#     keywords = " ".join(keywords) if keywords else ""
    
#     query = {
#         "query": {
#             "bool": {
#                 "must": [
#                     {
#                         "function_score": {
#                             "query": {
#                                 "match_all": {}
#                             },
#                             "functions": [
#                                 {
#                                     "script_score": {
#                                         "script": {
#                                             "source": """
#                                                 double vs_score = cosineSimilarity(params.queryVector, doc['para_vector']) + 
#                                                                 cosineSimilarity(params.queryVector, doc['title_vector']);
#                                                 if (params.keywords.length() > 0) {
#                                                     double keyword_score = doc['_score'].value;
#                                                     return vs_score + keyword_score * 0.3;
#                                                 }
#                                                 return vs_score;
#                                             """,
#                                             "params": {
#                                                 "queryVector": query_vectors,
#                                                 "keywords": keywords
#                                             }
#                                         }
#                                     }
#                                 }
#                             ],
#                             "boost_mode": "replace"
#                         }
#                     }
#                 ],
#                 "should": keywords [
#                     {
#                         "multi_match": {
#                             "query": keywords,
#                             "fields": ["title", "para"],
#                             "type": "best_fields"
#                         }
#                     }
#                 ]
#             }
#         }
#     }
#     return query

# def _get_context_from_db(query_vectors, keywords, k = 10, db_url="http://localhost:9200/legal_data/_search"):
#     headers = {"Content-Type": "application/json"}
#     post_query = search_query(query_vectors, keywords)
#     response = requests.post(db_url, json=post_query, headers=headers)
#     if response.status_code == 200:
#         hits = response.json().get('hits', {}).get('hits', [])[:k]
#         context = [
#             f"{hit.get('_source', {}).get('title', '无标题')} {hit.get('_source', {}).get('para', '无内容')}"
#             for hit in hits
#         ]
#     else:
#         context = "查询失败"
#     return context

# def rerank_documents(rewritten_query: str, context_hits: list[str], top_n, reranker: FlagReranker):
#     # combine the context hits and query.
#     sentence_pairs = [[rewritten_query, context] for context in context_hits]
#     # calculate the reranking score
#     scores = reranker.compute_score(sentence_pairs)

#     score_document = [{"score": score, "content": content} for score, content in zip(scores, context_hits)]
#     sorted_score_document = sorted(score_document, key=lambda x: x["score"], reverse=True)[:top_n]

#     return [doc["content"] for doc in sorted_score_document]

# # new test
# def get_context(models: dict, rewritten_query:str, keywords: list[str]):
#     embedding_model = models["embedding_model"]
#     reranker = models["rerank_model"]
#     vectors = get_query_vectors(embedding_model, rewritten_query)
#     context_hits = _get_context_from_db(vectors, keywords, 15)
#     reranked_context = rerank_documents(rewritten_query, context_hits, 10, reranker)

#     return reranked_context
####################

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

    return reranked_context

if __name__ == "__main__":
    config = Config()
    def embedding_models_init():
        embedding_model = FlagModel(
            model_name_or_path = 'BAAI/bge-large-zh-v1.5',
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True
        )

        rerank_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

        return {"embedding_model": embedding_model, "rerank_model": rerank_model}
    keywords = ["继承", "房产"]
    rewritten_query = "我奶奶20年前去世，然后叔叔去世，然后爸爸去世，最后爷爷去世，我可以继承父亲的房产吗"
    models = embedding_models_init()
    context = get_context(models, rewritten_query, keywords, config)
    print(context)

