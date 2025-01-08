import uuid
import json
from tqdm import tqdm
from splitter import *

def create_json_file_md(docs, output_path):
    json_data = []

    for doc in tqdm(docs):
        metadata = doc.metadata
        page_content = doc.page_content

        headers = [metadata.get("header1", ""), metadata.get("header2", ""), metadata.get("header3", "")]
        title = " ".join([header for header in headers if header])

        doc_id = str(uuid.uuid4())
        json_data.append({
            "id": doc_id,
            "title": title,
            "para": page_content
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def create_json_file_txt1(docs, output_path):
    json_data = []

    for doc in tqdm(docs):
        title = doc.metadata['title']
        doc_id = f"{str(uuid.uuid4())}-txt1"
        json_data.append({
            "id": doc_id,
            "title": title,
            "para": doc.page_content
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

# splite the data from the markdown type data files, and save then in to a json file.
def write_md():
    # process markdown
    text_splitter = LegalSplitterMD.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=20
    )
    data_path = "/home/jianglongyu/Documents/datasets/legal_dataset/legal_book_md"
    docs = LawLoader(data_path).load_and_split(text_splitter=text_splitter)
    output_path = "legal_data_md.json"
    create_json_file_md(docs, output_path)

# splite the data from the txt type data files, and save then in to a json file.
def write_txt1():
    path_to_law_txt = "/home/jianglongyu/Documents/datasets/legal_dataset/legal_book"
    law_loader = LawLoaderTXT(path=path_to_law_txt)
    splitter = LegalSplitterTXT(chunk_size=300, chunk_overlap=20)
    split_documents = splitter.load_and_split(law_loader)

    output_path = "legal_data_txt1.json"
    create_json_file_txt1(split_documents, output_path)

# splite the data from the txt type data file (another one), and save then in to a json file.
def write_txt2():
    path_to_law_txt1 = "/home/jianglongyu/Documents/datasets/legal_dataset/legal_article/article.txt"
    docs = legal_splitter_txt2(path_to_law_txt1)

    output_path = "legal_data_txt2.json"
    json_data = []
    for doc in tqdm(docs):
        doc_id = f"{str(uuid.uuid4())}-txt2"
        json_data.append({
            "id": doc_id,
            "title": doc.metadata['title'],
            "para": doc.page_content
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    write_txt1()
    pass