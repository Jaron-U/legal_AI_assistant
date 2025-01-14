from summarizer import Summarizer
from duckduckgo_search import DDGS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from retrieve import rerank_documents
from FlagEmbedding import FlagReranker

def web_retrieve(keywords: List, rewritten_query: str,
                reranker: FlagReranker, 
                content_summary_model: Summarizer,
                max_results=10, ranked_k=5,
                chunk_size=300, chunk_overlap=20):
    combined_query = " ".join(keywords)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    
    search_results = DDGS().text(
        keywords=combined_query,
        region='wt-wt',
        safesearch='off',
        max_results=max_results
    )

    combined_text = "\n".join(
        [result.get('body', '') for result in search_results if 'body' in result]
    )

    split_docs = text_splitter.split_text(combined_text)

    reranked_docs = rerank_documents(rewritten_query, split_docs, 
                                     top_n=ranked_k, reranker=reranker)
    reranked_docs = "\n\n".join(reranked_docs)
    summary = content_summary_model.get_summary(query=reranked_docs)
    return summary