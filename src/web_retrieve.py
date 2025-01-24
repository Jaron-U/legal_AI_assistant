from duckduckgo_search import DDGS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from FlagEmbedding import FlagReranker

from src.summarizer import Summarizer
from src.config import Config
from src.utils import rerank_docs

def retrieve_from_web(db_content_check_res:str, 
                      summarizer: Summarizer, query_dict: Dict,
                      reranker: FlagReranker, config: Config,
                      chunk_size=300, chunk_overlap=20):
    if "yes" in db_content_check_res.lower():
        return None
    else:
        print("Retrieving from web...")
        combined_query = " ".join(query_dict['keywords'])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap)
        search_results = DDGS().text(
            keywords=combined_query,
            region='wt-wt',
            safesearch='off',
            max_results=config.retrieved_web_top_k
        )

        combined_text = "\n".join(
            [result.get('body', '') for result in search_results if 'body' in result]
        )
        split_docs = text_splitter.split_text(combined_text)

        reranked_docs = rerank_docs(query_dict['rewritten'], split_docs, reranker, config.ranked_web_top_k)

        reranked_docs = "\n".join(reranked_docs)
        summary = summarizer.get_summary(reranked_docs)
        return summary