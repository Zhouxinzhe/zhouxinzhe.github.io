---
layout:       post
title:        "调用硅基流动API"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - ChatGPT
---

说来惭愧，大语言模型火了这么些年，一直是在网页端使用，仅当作回答问题的工具，不曾用api调用。趁着这一波DeepSeek的热潮，赶紧尝试一下大模型的api调用，未未来进一步的集成开发打一个基础。

### 硅基流动

这里使用的是硅基流动平台的api服务。硅基流动（SiliconFlow）是一家专注于生成式人工智能（GenAI）计算基础设施的平台，致力于通过技术创新降低大模型（如生成式AI和大语言模型）的部署和推理成本。该平台提供了api的服务，且提供多种大语言模型及其他生成式AI模型。

### 操作步骤
1. [新建密钥](https://cloud.siliconflow.cn/account/ak)

2. [查阅API手册](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)

3. python调用api：

   ```python
   import requests
   
   url = "https://api.siliconflow.cn/v1/chat/completions"
   
   payload = {
       "model": "deepseek-ai/DeepSeek-V2.5",
       "messages": [
           {
               "role": "user",
               "content": "你好"
           }
       ],
       "stream": False
   }
   headers = {
       "Authorization": "Bearer <key>",
       "Content-Type": "application/json"
   }
   
   response = requests.request("POST", url, json=payload, headers=headers)
   
   print(response.text)
   ```

   result：

   ```python
   {"id":"01950de7a147eb74dc319004a90dbe46","object":"chat.completion","created":1739695039,"model":"deepseek-ai/DeepSeek-V2.5","choices":[{"index":0,"message":{"role":"assistant","content":"你好！如何帮到你？"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":7,"total_tokens":11},"system_fingerprint":""}
   ```

   