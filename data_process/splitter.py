from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.docstore.document import Document
from typing import Any, Iterable, List, Dict
from .utils import *
import re, os, json

class LegalSplitterMD(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        separators = [r"第\S*条"]
        is_separator_regex = True

        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        super().__init__(separators=separators, is_separator_regex=is_separator_regex, **kwargs)
    
    def split_documents(self, documents:Iterable[Document]) -> List[Document]:
        texts, metadatas = [], []
        for doc in documents:
            md_docs = self.md_splitter.split_text(doc.page_content)
            for md_doc in md_docs:
                texts.append(md_doc.page_content)
                metadatas.append(md_doc.metadata | doc.metadata | {"book": md_doc.metadata.get("header1")})
        
        return self.create_documents(texts, metadatas=metadatas)

class LegalSplitterTXT(RecursiveCharacterTextSplitter):
    chapter_pattern = r"(第[一二三四五六七八九十百]+章[\u4e00-\u9fa5]*)"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
    
    def extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Extract chapters and their start positions."""
        matches = re.finditer(self.chapter_pattern, text)
        chapters = []
        for match in matches:
            chapters.append({"title": match.group(), "start": match.start()})
        return chapters
    
    def assign_titles(self, text: str, chunks: List[str], chapters: List[Dict[str, Any]], filename: str) -> List[Document]:
        """Assign chapter titles to each chunk and include filename."""
        documents = []
        for chunk in chunks:
            chunk_start = text.find(chunk) # Find the start position of the chunk
            chapter_title = next(
                (chapter["title"] for chapter in reversed(chapters) if chapter["start"] <= chunk_start),
                None
            )
            if chapter_title is not None:
                full_title = f"{filename} {chapter_title}" # Combine filename and chapter title
            else:
                full_title = filename
            documents.append(Document(page_content=chunk, metadata={"title": full_title}))
        return documents

    def split_text_with_titles(self, text: str, filename: str) -> List[Document]:
        """Split text and assign titles including filename."""
        chapters = self.extract_chapters(text)
        chunks = self.split_text(text)
        return self.assign_titles(text, chunks, chapters, filename)
    
    def load_and_split(self, loader: LawLoaderTXT) -> List[Document]:
        """Load documents using LawLoaderTXT and split with titles."""
        docs = loader.load()
        all_documents = []
        for doc in docs:
            text = doc.page_content
            filename = os.path.basename(doc.metadata["source"]).replace(".txt", "")  # Extract filename without extension
            documents = self.split_text_with_titles(text, filename)
            all_documents.extend(documents)
        return all_documents
    
def legal_splitter_txt2(file_path, chunk_size=300, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {line}")
                continue

            answer = data.get("answer", "")
            if not answer:
                continue
            
            try:
                title, content = answer.strip().split(" ", 1)
            except ValueError:
                title = "法律条例"
                content = answer.strip()

            chunks = splitter.split_text(content)

            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"title": title}))
    return documents

if __name__ == "__main__":
    # # for markdown data
    # text_splitter = LegalSplitterMD.from_tiktoken_encoder(
    #     chunk_size=300, chunk_overlap=20
    # )
    # data_path_md = "/home/jianglongyu/Documents/datasets/legal_dataset/legal_book_md"
    # docs = LawLoader(data_path_md).load_and_split(text_splitter=text_splitter)

    # for legal_book txt file data
    path_to_law_txt = "/home/jianglongyu/Documents/datasets/legal_dataset/legal_book"
    law_loader = LawLoaderTXT(path=path_to_law_txt)
    splitter = LegalSplitterTXT(chunk_size=300, chunk_overlap=20)
    split_documents = splitter.load_and_split(law_loader)

    for doc in split_documents[:5]: 
        print(f"Title: {doc.metadata['title']}")
        print(f"Content: {doc.page_content}") 
        print()
