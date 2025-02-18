---
layout:       post
title:        "【动手学Langchain】1-初识Langchain与环境安装"
author:       "Orchid"
header-style: text
catalog:      true
hidden:       false
tags:
    - Langchain
    - LLM
---

# 初识 Langchain

Langchain，一个能够轻松、高效地构建 `LLM` 应用的工具。

## 为什么需要 Langchain

Langchain 是一个用于开发LLM应用的集成架构，它为开发者提供了一系列的工具和组件，使得与语言模型中的各种数据的连接、应用和优化变得简单直接。

同时 Langchain 在开发者社区中的受欢迎程度逐渐上升，表现出了极大的实用性和潜力。

## Langchain 的功能特点

* **简洁性**。开发者只需要几行代码就能构建一个`LLM` 程序
* **为开发者提供了丰富的内置链组件**。解决了重复编写代码的问题，简化了开发流程。
* **可以实现RAG技术**，实现LLM与真实世界的在线数据增强。
* **提示词模板**。可以使用官方提供的，也可以自定义创建，实现与各种应用和工具的紧密集成。

## Langchain 的六大核心模块

1. **模型 I/O（Model I/O）**
   - **功能**：标准化语言模型的输入和输出，包括输入提示的构建、模型调用以及输出的解析。LangChain 提供了提示模板（Prompt Template）来生成高质量的输入提示，支持动态参数替换和格式校验。此外，通过输出解析器，可以将模型的非结构化输出转换为结构化数据，方便后续处理。
2. **链（Chains）**
   - **功能**：将多个操作步骤串联成复杂的工作流，支持动态适应输入内容。链是 LangChain 中用于构建多步骤任务的核心模块，可以将语言模型调用、工具操作等组合在一起，实现自动化流程。
3. **数据增强（Data Augmentation）**
   - **功能**：通过检索外部数据（如文档、数据库）并将其传递到语言模型中，增强模型的输出质量。这一模块通常结合检索增强生成（Retrieval-Augmented Generation, RAG）技术，提升回答的准确性和相关性。
4. **记忆（Memory）**
   - **功能**：用于存储和管理对话上下文，支持短期记忆（会话内存储）和长期记忆（跨会话存储）。记忆模块可以帮助模型更好地理解上下文信息，从而生成更连贯的对话。
5. **代理（Agents）**
   - **功能**：代理通过语言模型动态选择和执行一系列操作，以完成复杂任务。代理的核心是工具（Tools）的调用，这些工具可以执行特定功能（如数据库查询、API 调用等）。代理的工作流程包括输入理解、计划制定、工具调用、结果整合和反馈循环。
6. **回调处理器（Callbacks）**
   - **功能**：回调处理器是 LangChain 的回调机制的核心组件，用于在任务执行过程中监听特定事件并触发预定义的操作。例如，可以用于日志记录、监控、流式传输等。

这些模块共同构成了 LangChain 的核心架构，使得开发者能够灵活地构建复杂的语言处理应用。



# 开发环境搭建

1. **创建虚拟环境**

   为了避免与系统 Python 环境冲突，建议创建一个虚拟环境。这里使用 `conda`：

   ```bash
   conda create -n langchain python=3.9
   conda activate langchain
   ```

2. **安装 Langchain**

   安装 LangChain 的核心库和社区扩展库：

   ```bash
   pip install langchain
   pip install langchain-community
   ```

   如果使用第三方封装的大语言模型，如通义千问，需要安装额外依赖，不同的大语言模型需要不同的依赖。通义千问：

   ```bash
   pip install dashscope
   ```

   如果需要安装特定的集成包，如 Deepseek，可以安装额外依赖：

   ```bash
   pip install langchain-deepseek
   ```

3. **配置 API 密钥**

   * 通义千问 

     * API 密钥获取：[阿里云百炼](https://bailian.console.aliyun.com/#/home)

     * API 配置：

       ```python
       import os
       os.environ["DASHSCOPE_API_KEY"] = 'sk-xx'
       ```

   * Deepseek

     * API 密钥获取：[DeepSeek 开放平台](https://platform.deepseek.com/usage)

     * API 配置：

       ```python
       import os
       os.environ["DEEPSEEK_API_KEY"] = 'sk-xx'
       ```

4. **验证安装**

   * 通义千问

     [模型列表_大模型服务平台百炼(Model Studio)-阿里云帮助中心](https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.help-menu-2400256.d_0_2.5f031d1ckndWLR&scm=20140722.H_2840914._.OR_help-T_cn~zh-V_1)

     ```python
     from langchain_community.chat_models import ChatTongyi
     import os
     os.environ["DASHSCOPE_API_KEY"] = 'sk-xx'
     
     tongyi_chat = ChatTongyi(
         model="qwen-max",
     )
     
     messages = [
         ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
         ("human", "我喜欢编程。"),
     ]
     response = tongyi_chat.invoke(messages)
     
     print(response.content)
     ```

   * Deepseek

     ```python
     from langchain_deepseek import ChatDeepSeek
     import os
     os.environ["DEEPSEEK_API_KEY"] = 'sk-xx'
     
     deepseek_chat = ChatDeepSeek(
         model="deepseek-chat",
     )
     
     messages = [
         ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
         ("human", "我喜欢编程。"),
     ]
     response = tongyi_chat.invoke(messages)
     
     print(response.content)
     ```

     

