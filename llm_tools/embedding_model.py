from openai import OpenAI
import configparser
import numpy as np

class EmbeddingModel:
    def __init__(self, embedding_model, config_file='./llm_tools/configs/llm.ini'):
        self.embedding_model = embedding_model
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.local_api_key = self.config[embedding_model]['local_api_key']
        self.local_base_url = self.config[embedding_model]['local_base_url']
        self.client = OpenAI(api_key=self.local_api_key, base_url=self.local_base_url)
            
    def embed_query(self, query=None):
        response = self.client.embeddings.create(
            input = query,
            model = self.embedding_model
        )
        return response.data[0].embedding
    
    def embed_documents(self, documents=None):
        emb_list = []
        response = self.client.embeddings.create(
            input = documents,
            model = self.embedding_model
        )
        for i in range(len(response.data)):
            emb_list.append(response.data[i].embedding)
        return emb_list

# 使用範例
if __name__ == '__main__':
    embmodel = EmbeddingModel(embedding_model="m3e-base")
    query = "The food was delicious and the waiter was friendly."
    query_embedding = np.array(embmodel.embed_query(query))
    print(len(query_embedding))
    # documents = ["The food was delicious and the waiter was friendly.",
    #              "The service was slow and the food was not very good."]
    # document_embeddings = np.array(embmodel.embed_documents(documents))
    # print(document_embeddings.shape)