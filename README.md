# LLM工具說明
## llm_tools模組說明
### 參數設定
- 查看configs/llm.ini
    - **DEFAULT**: 預設的參數，包含temperature、max_tokens、top_p、frequency_penalty、presence_penalty等
    - **Azure OpenAI系列**: 設定azure_api_base、azure_api_key、azure_api_version，打了會花錢，請謹慎使用
    - **地端LLM/Embedding Model系列**: 設定local_api_key、local_base_url，目前統一接口8887，若有實驗新模型可自行新增
  
### 導入方法
- 第一步：將llm_tools資料夾放在跟操作程式碼相同路徑
- 第二步：在python中導入以下模組
    - from llm_tools.llm_chat import LLMChat
    - from llm_tools.embedding_model import EmbeddingModel

### 函數內容
- llm_chat.py 重要參數與功能
    - params (dictionary): 可控制參數
    - response_format (string): 可設定"json_object"，讓LLM返回json物件，但是prompt內要有json的說明
    - stream (bool): 設定為True可使用流式輸出
    - history (list): OpenAI接口回傳的history，格式為[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}. {"role": "system", "content": completion}]
- embedding_model.py 重要參數與功能
    - 單句轉Embedding: 輸出為array
    - 多句(list)轉Embedding: 會集合成batch方式轉譯，跑批的時候速度較快，輸出為array

### 範例程式碼
細節參照使用說明tutorial.py或tutorial.ipynb
```python
from llm_tools.llm_chat import LLMChat
from llm_tools.embedding_model import EmbeddingModel

# 設定參數
params = {
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# 建立LLMChat物件
llmchat = LLMChat(model="Qwen2-7B-Instruct")

# 設定prompt
prompt = "請問你是誰？"

# 取得response
history = None
response, history = llmchat.chat(query=query, system=system, stream=True)

# 取得embedding
embedding_model = EmbeddingModel()
embedding = embedding_model.get_embedding(prompt)

# 取得多句embedding
embedding_list = ["你好", "早上好", "晚上好"]
embedding_batch = embedding_model.get_embedding_batch(embedding_list)
```
