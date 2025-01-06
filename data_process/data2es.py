from elasticsearch import Elasticsearch, helpers
import json, os, uuid
from FlagEmbedding import FlagModel
es = Elasticsearch('http://localhost:9200')
from datetime import datetime, timezone
from tqdm import tqdm

def create_db(index_name="legal_data"):
    create_index_body = {
        "settings": {
            "index": {
                "number_of_shards": 3,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text"
                },
                "para": {
                    "type": "text"
                },
                "doc_type": {
                    "type": "keyword"
                },
                "doc_id": {
                    "type": "keyword"
                },
                "create_time": {
                    "type": "date",
                    "format": "yyyy-MM-dd HH:mm:ss"
                },
                "title_vector": {
                    "type": "dense_vector",
                    "dims": 1024
                },
                "para_vector": {
                    "type": "dense_vector",
                    "dims": 1024
                }
            }
        }
    }

    # if the db is exit just delete it
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=create_index_body)

    # if not es.indices.exists(index=index_name):
    #     es.indices.create(index=index_name, body=create_index_body)

def load_json_files(dataset_path):
    data_list = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(dataset_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data_list.extend(json.load(file))
    return data_list

def generate_vectors(model: FlagModel, title, para):
    title_vector = model.encode(title)
    para_vector = model.encode(para)
    return title_vector, para_vector

def create_es_action(index_name, record, title_vector, para_vector):
    action = {
        "_index": index_name, 
        "_id": str(uuid.uuid4()), 
        "_source": {
            "title": record["title"],
            "para": record["para"],
            "doc_id": record["id"],
            "create_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "title_vector": title_vector,
            "para_vector": para_vector,
        },
    }

    return action

def data2es(index_name, embed_model, dataset_path):
    data = load_json_files(dataset_path)
    actions = []
    for record in tqdm(data):
        title_vector, para_vector = generate_vectors(embed_model, record["title"], record["para"])
        action = create_es_action(index_name, record, title_vector, para_vector)
        actions.append(action)
    helpers.bulk(es, actions)
    print(f"write in {len(actions)} data into es")

if __name__ == "__main__":
    model = FlagModel('BAAI/bge-large-zh-v1.5', 
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=True)
    index_name = "legal_data"
    dataset_path = "dataset"
    create_db(index_name)
    data2es(index_name, model, dataset_path)



    