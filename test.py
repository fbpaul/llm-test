import numpy as np

from llm_tools.embedding_model import EmbeddingModel
from llm_tools.llm_chat import LLMChat

## Test LLM
# llmchat = LLMChat(model="gpt-4o")
# llmchat = LLMChat(model="Qwen1.5-14B-Chat")
llmchat = LLMChat(model="Qwen2-7B-Instruct")

params = {
    "temperature": 0.8,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 1.4,
    "presence_penalty": 0
}
try:
    system = """
    任務：解讀主題與內容
    請從下面的文字中解讀該段落的語意主題與內容，應該詳細閱讀資訊，並提取其語意主題與描述內容。
    """
    query = """
    咱們的護國神山，台積電在今年四月的北美技術論壇上，發表埃米級的 A16 製程，預計於 2026 年開始量產，引發各大科技龍頭的瘋搶。
    台積的 A16 製程更引入 SPR 系統，不只減少了 IR 降壓，相較於過去的 N2P 製程，能在相同性能下，減少最多 20% 的功耗，同時提升近 10% 的晶片密度。
    OpenAI 也希望得到 A16 製程的協助，打造屬於自家的 AI 晶片，用以強化旗下的 GPT 語言模型，以及影像生成模型 Sora。
    """
    history = None
    # response, history = llmchat.chat(query=query, system=system, params=params, response_format="json_object", stream=True) # response規定json
    response, history = llmchat.chat(query=query, system=system, stream=True)
    print()
    print(response)
except ValueError as e:
    print(e)

## Test LLM Chatbot
# history=None 
# is_stream=True
# while True:
#     input_text = input("請輸入你的問題: ")  
#     if input_text.lower() == "quit":  
#         break
#     response, history = llmchat.chat(query=input_text, history=history, stream=is_stream)
#     if is_stream==True:
#         print()
#     else:
#         print(response)


## Test EmbeddingModel
# embmodel = EmbeddingModel(embedding_model="m3e-base")
# query = "The food was delicious and the waiter was friendly."
# query_embedding = np.array(embmodel.embed_query(query))
# print(len(query_embedding))
# documents = ["The food was delicious and the waiter was friendly.",
#              "The service was slow and the food was not very good."]
# document_embeddings = np.array(embmodel.embed_documents(documents))
# print(document_embeddings.shape)