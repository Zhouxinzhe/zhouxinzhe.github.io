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

### 基本操作步骤

1. [新建密钥](https://cloud.siliconflow.cn/account/ak)

2. [查阅API手册](https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions)

3. python调用api：

   ```python
   import requests
   
   url = "https://api.siliconflow.cn/v1/chat/completions"
   
   payload = {
       "model": "deepseek-ai/DeepSeek-V3",
       "messages": [
           {
               "role": "user",
               "content": "中国大模型行业2025年将会迎来哪些机遇和挑战？"
           }
       ],
       "stream": False,
       "max_tokens": 512,
       "stop": ["null"],
       "temperature": 0.7,
       "top_p": 0.7,
       "top_k": 50,
       "frequency_penalty": 0.5,
       "n": 1,
       "response_format": {"type": "text"},
       "tools": [
           {
               "type": "function",
               "function": {
                   "description": "<string>",
                   "name": "<string>",
                   "parameters": {},
                   "strict": False
               }
           }
       ]
   }
   headers = {
       "Authorization": "Bearer <token>",
       "Content-Type": "application/json"
   }
   
   response = requests.request("POST", url, json=payload, headers=headers)
   
   print(response.text)
   ```

### 请求参数说明

| 参数                | 类型    | 必填 | 描述                                                         |
| :------------------ | :------ | :--- | :----------------------------------------------------------- |
| `model`             | String  | 是   | 指定要使用的模型版本。例如：`deepseek-ai/DeepSeek-V3`。      |
| `messages`          | List    | 是   | 定义用户与模型之间的对话内容。包含角色（`role`）和内容（`content`）。 |
| `stream`            | Boolean | 否   | 是否启用流式响应。`True` 表示流式输出，`False` 表示非流式输出。默认为 `False`。 |
| `max_tokens`        | Integer | 否   | 设置模型生成的最大 token 数量。用于限制生成内容的长度。默认值可能因模型而异。 |
| `stop`              | List    | 否   | 定义生成停止的条件。当生成内容包含指定的字符串时，生成过程会结束。 |
| `temperature`       | Float   | 否   | 控制生成内容的随机性。值越低，生成内容越确定；值越高，生成内容越随机。默认值通常为 `0.7`。 |
| `top_p`             | Float   | 否   | 控制生成内容的多样性，表示模型在生成时考虑的候选 token 的累积概率。默认值通常为 `0.7`。 |
| `top_k`             | Integer | 否   | 控制生成内容的多样性，表示模型在生成时考虑的候选 token 数量。默认值可能因模型而异。 |
| `frequency_penalty` | Float   | 否   | 对重复内容的惩罚系数。值越高，模型越倾向于生成不重复的内容。默认值通常为 `0.5`。 |
| `n`                 | Integer | 否   | 返回的生成结果数量。默认值为 `1`。                           |
| `response_format`   | Dict    | 否   | 定义返回内容的格式，例如 `{"type": "text"}`。                |
| `tools`             | List    | 否   | 扩展功能或插件的定义，具体功能需参考 API 文档。              |

### 返回参数说明

| 参数               | 类型   | 描述                                                         |
| :----------------- | :----- | :----------------------------------------------------------- |
| id                 | 字符串 | 用于唯一标识聊天完成的标识符。                               |
| choices            | 数组   | 包含生成内容的列表。如果`n`大于1，则可以有多个选择。         |
| created            | 整数   | 聊天完成创建的Unix时间戳（以秒为单位）。                     |
| model              | 字符串 | 用于生成内容的模型。                                         |
| system_fingerprint | 字符串 | 此指纹表示模型运行时的后端配置。可与seed请求参数一起使用，了解可能影响确定性的后端更改。 |
| usage              | 对象   | 完成请求的使用统计信息。                                     |
| finish_reason      | 字符串 | 表示聊天完成的原因。可能的值包括"stop"（API返回了完整的聊天完成而没有受到任何限制），“length”（生成超过了max_tokens或对话超过了max context length），等等。 |

### Example

code：

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
    "Authorization": "Bearer sk-vnfrphkwzplvwqdxiszwoqbooofhlicfhvuvkfnyqluaqvdl",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)
```

result：

```
{
"id":"01950de7a147eb74dc319004a90dbe46",
"object":"chat.completion",
"created":1739695039,
"model":"deepseek-ai/DeepSeek-V2.5",
"choices":[{"index":0,"message":{"role":"assistant","content":"你好！如何帮到你？"},"finish_reason":"stop"}],
"usage":{"prompt_tokens":4,"completion_tokens":7,"total_tokens":11},
"system_fingerprint":""
}
```

