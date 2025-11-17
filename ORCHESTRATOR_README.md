# Orchestrator Agent

这是一个顶层的编排代理（Orchestrator Agent），它可以根据用户的需求自动选择并调用适当的子代理来完成任务。

## 功能

编排代理负责分析用户的请求，并根据请求类型将任务分派给以下两个专门的子代理之一：

1. **Coder Agent** - 负责处理与编程和软件开发相关的任务
   - 编写和分析代码
   - 调试和测试
   - 代码优化和重构
   - 执行代码
   - 软件架构问题

2. **File Manager Agent** - 负责处理文件和文档管理相关的任务
   - 搜索文件和目录
   - 读取文件内容
   - 写入和创建文件
   - 管理文件系统
   - 文档分析

## 安装和运行

### 前置要求

- Python 3.8+
- 安装所需的依赖包（见requirements.txt）

### 环境配置

1. 复制`.env.example`文件（如果存在）为`.env`并配置以下变量：

   对于Ollama模型：
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1:8b
   ```

   对于云API模型：
   ```
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 运行编排代理

```bash
python orchestrator_main.py
```

## 使用方法

1. 启动程序后，系统会询问您是否要开始一个新的对话：
   - 输入`y`或`yes`开始一个新对话（将清除之前的记忆）
   - 输入`n`或`no`继续之前的对话

2. 选择要使用的模型类型：
   - 输入`1`选择Ollama本地模型
   - 输入`2`选择云API模型

3. 输入您的请求，编排代理将自动分析并选择适当的子代理来处理您的请求。

4. 输入`exit`、`quit`或`bye`退出程序。

## 示例

### 编程相关请求（将使用Coder Agent）

```
> 请帮我写一个Python函数来计算斐波那契数列
```

### 文件管理相关请求（将使用File Manager Agent）

```
> 请帮我查找当前目录下所有的.txt文件
```

## 文件结构

```
.
├── orchestrator_agent.py  # 编排代理的主要实现
├── orchestrator_main.py   # 运行编排代理的入口文件
├── coder_agent.py         # Coder Agent的实现
├── agent.py               # File Manager Agent的实现
├── tools/                 # 各种工具的实现
└── README.md              # 本文件
```

## 工作流程

1. 用户输入请求
2. 编排代理分析请求类型
3. 根据分析结果选择适当的子代理
4. 将请求分派给选定的子代理
5. 子代理处理请求并返回结果
6. 编排代理将结果呈现给用户

## 注意事项

- 确保已正确配置环境变量
- 如果使用Ollama，请确保Ollama服务正在运行
- 如果使用云API，请确保API密钥有效且有足够的配额