import requests
import transformers
transformers.logging.set_verbosity_error()
from FlagEmbedding import FlagModel, FlagReranker
from typing import Dict, List

from src.config import Config
from src.utils import rerank_docs

def create_search_query(
    question_vector: List[float],      
    keyword_vectors: Dict[str, list[float]], 
    question_weight: float = 1.0,          
    keyword_weight: float = 1.5,      
    top_k: int = 15                       
) -> Dict:
    query_conditions = []
    
    # 1. query vector search
    query_conditions.append({
        "script_score": {
            "script": {
                "source": f"""
                    double paraScore = cosineSimilarity(params.queryVector, doc['para_vector']) + 1.0;
                    double titleScore = cosineSimilarity(params.queryVector, doc['title_vector']) + 1.0;
                    return (paraScore + titleScore) * {question_weight};
                """,
                "params": {
                    "queryVector": question_vector
                }
            }
        }
    })
    
    # 2. keyword vector search
    for _, keyword_vector in keyword_vectors.items():
        query_conditions.append({
            "script_score": {
                "script": {
                    "source": f"""
                        double paraScore = cosineSimilarity(params.queryVector, doc['para_vector']) + 1.0;
                        double titleScore = cosineSimilarity(params.queryVector, doc['title_vector']) + 1.0;
                        return (paraScore + titleScore) * {keyword_weight};
                    """,
                    "params": {
                        "queryVector": keyword_vector
                    }
                }
            }
        })
    
    # Combine all conditions using `function_score`
    query = {
        "query": {
            "function_score": {
                "query": {"match_all": {}},
                "functions": query_conditions,
                "score_mode": "sum",    
                "boost_mode": "sum"
            }
        },
        "size": top_k
    }
    
    return query

def _get_query_vectors(model: FlagModel, rewrite_query: str) -> list[float]:
    vectors = model.encode(rewrite_query).tolist()
    return vectors

def _get_keyword_vectors(model: FlagModel, keywords: list[str]) -> dict[str, list[float]]:
    keyword_vectors = {}
    for keyword in keywords:
        keyword_vectors[keyword] = model.encode(keyword).tolist()
    return keyword_vectors

def _get_content_from_db(query_vector: List[float], keyword_vectors: Dict[str, list[float]], 
                         retrieved_top_k: int = 15, db_url="http://localhost:9200/legal_data/_search"
                         ) -> List[str]:
    search_query = create_search_query(query_vector, keyword_vectors, top_k=retrieved_top_k)
    response = requests.post(db_url, json=search_query)
    if response.status_code == 200:
        hits = response.json().get('hits', {}).get('hits', [])
        context_docs = [
            f"标题：{hit.get('_source', {}).get('title', '无标题')}\n内容：{hit.get('_source', {}).get('para', '无内容')}"
            for hit in hits
        ]
    else:
        context_docs = None
    
    return context_docs
    
def retrieve_from_db(query_dict: Dict, models: Dict, config: Config) -> List[str]:
    embedding_model: FlagModel = models["embedding_model"]
    reranker: FlagReranker = models["rerank_model"]
    query_vector = _get_query_vectors(embedding_model, query_dict["rewritten"])
    keyword_vectors = _get_keyword_vectors(embedding_model, query_dict["keywords"])
    context_docs = _get_content_from_db(query_vector, keyword_vectors, config.retrieved_db_top_k, config.db_url)
    if context_docs is None:
        return None
    reranked_docs = rerank_docs(query_dict["rewritten"], context_docs, reranker, config.ranked_db_top_k)

    return reranked_docs

