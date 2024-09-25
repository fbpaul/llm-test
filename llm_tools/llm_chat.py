import json
import torch
from openai import AzureOpenAI, OpenAI
import sys
import configparser
import os

class LLMChat:
    def __init__(self, model, config_file='./llm_tools/configs/llm.ini'):
        self.model = model
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        if 'gpt' in model:
            self.azure_api_base = self.config[model]['azure_api_base']
            self.azure_api_key = self.config[model]['azure_api_key']
            self.azure_api_version = self.config[model]['azure_api_version']
            self.client = AzureOpenAI(azure_endpoint=self.azure_api_base, api_key=self.azure_api_key, api_version=self.azure_api_version)
        else:
            self.local_api_key = self.config[model]['local_api_key']
            self.local_base_url = self.config[model]['local_base_url']
            self.client = OpenAI(api_key=self.local_api_key, base_url=self.local_base_url)
            
    def initialize_history(self, system_message, user_message):
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
    def chat(self, query, history=None, system="You are a helpful assistant.", params=None, response_format=None, stream=False):
        torch.cuda.empty_cache()    
        if history is None:
            history = self.initialize_history(system, query)
        else:
            history.append({"role": "user", "content": query})
        if params == None:
            params = {
                "temperature": self.config['DEFAULT'].getfloat('temperature'),
                "max_tokens": self.config['DEFAULT'].getint('max_tokens'),
                "top_p": self.config['DEFAULT'].getfloat('top_p'),
                "frequency_penalty": self.config['DEFAULT'].getfloat('frequency_penalty'),
                "presence_penalty": self.config['DEFAULT'].getfloat('presence_penalty')
            }
        else:
            params = params
        if response_format==None:
            completion = self.client.chat.completions.create(
            model=self.model,
            stream=stream,
            messages=history,
            **params
        )
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": response_format},
                stream=stream,
                messages=history,
                **params
            )
        if stream:
            response = ""
            for chunk in completion:
                try:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        response += content
                except:
                    pass
        else:
            response = completion.choices[0].message.content

        history.append({"role": "system", "content": response})
        
        return response, history
 

# 使用範例
if __name__ == '__main__':
    # llmchat = LLMChat(model="gpt-4o")
    # llmchat = LLMChat(model="qwen2")
    # llmchat = LLMChat(model="Qwen1.5-14B-Chat")
    llmchat = LLMChat(model="Qwen2-7B-Instruct")
    # llmchat = LLMChat(model="Qwen1.5-14B-Chat")
    
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
        response, history = llmchat.chat(query=query, system=system, params=params, response_format="json_object", stream=True) # response規定json
        # response, history = llmchat.chat(query=query, system=system, stream=True)
        print()
        print(response)
    except ValueError as e:
        print(e)
    
    
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