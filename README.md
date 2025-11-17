# MyAgent - Multi-Agent AI System

A powerful interactive AI multi‑agent system built with LangGraph, LangChain, and Ollama/OpenAI APIs.  
The project provides a terminal-based orchestrator with specialized sub‑agents (coder, file manager, log analyzer), local tools, and optional MCP (Model Context Protocol) integrations.

![LangGraph Workflow](orchestrator_workflow.png)

## 🌟 Key Features

- **Multi-Agent Architecture**: Orchestrator, Coder, File Manager, and Log Analyzer agents that automatically select and collaborate based on task type
- **ReAct-Based Reasoning**: All agents follow a ReAct loop (Think → Act via tools → Observe → Think → Answer) for more reliable step‑by‑step problem solving
- **Execution Planning & Confirmation**: Orchestrator creates a multi‑step plan, presents it for your confirmation/modification, and only then executes it
- **Inter-Agent Collaboration**: Sub‑agents can call each other (e.g., Log Analyzer asks Coder to generate a PowerShell fix script, Coder asks File Manager to open files)
- **Multiple Model Support**: Choose between local Ollama models or cloud-based OpenAI‑compatible APIs (configured in `config.json` + `.env`)
- **Rich Terminal UI**: Beautiful console interface with Markdown rendering and syntax highlighting
- **Configurable Working Directory**: All agents operate inside a centralized working folder configured via `config.json` (e.g. `E:\AfterMars\Agent_Working_Place`)
- **Local Tools**: File operations, document reading, PDF processing, code analysis/execution, log analysis, and unit test execution
- **MCP Integration**: Support for remote MCP servers including Desktop Commander, Python sandbox, DuckDuckGo search, and GitHub
- **Persistent Memory**: SQLite-based checkpoint system for conversation continuity (per‑agent databases in the working directory)
- **Workflow Visualization**: Automatic generation of Mermaid diagrams for the agents’ workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- uv (Python package manager)
- Docker (for MCP Docker images)
- Ollama (for local models) or API keys for cloud services

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MyAgent
   ```

2. **Initialize the uv workspace**
   ```bash
   uv init
   ```

3. **Install dependencies**
   ```bash
   uv add -r requirements.txt
   ```

4. **Sync the environment**
   ```bash
   uv sync
   ```

5. **Activate the virtual environment**
   ```bash
   # Windows
   .\.venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

### Configuration

Create a `.env` file in the project root:

```env
# For Ollama (local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# For Cloud API (OpenAI-compatible)
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4

# For GitHub MCP integration
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_token_here
```

### Running the Agents

> **Recommended entrypoint:** Orchestrator Agent – it will route tasks to the appropriate sub‑agent.

#### File Manager Agent (Base / Single-Agent Mode)
```bash
# Using uv
uv run file_manager_main.py

# Or with Python directly (after activating venv)
python file_manager_main.py
```

#### Orchestrator Agent (Recommended)
```bash
# Using uv
uv run orchestrator_main.py

# Or with Python directly (after activating venv)
python orchestrator_main.py
```

#### Coder Agent
```bash
# Using uv
uv run coder_main.py

# Or with Python directly (after activating venv)
python coder_main.py
```

> There is also a **Log Analyzer Agent** entrypoint if you want to work only with logs:
> ```bash
> uv run log_analyzer_main.py
> # or
> python log_analyzer_main.py
> ```

## 🛠️ Agent Types

### 1. Orchestrator Agent

Top‑level coordinator that analyzes your request, creates an execution plan, asks for confirmation, and then routes to/coordinates sub‑agents.

**Features**:
- Analyzes user request types and selects appropriate sub‑agents (`coder`, `file_manager`, `log_analyzer`)
- Creates a ReAct‑style execution plan and waits for user confirmation/modification
- Coordinates work and information flow between multiple agents
- Maintains per‑agent conversation memory (stored in per‑agent SQLite checkpoint DBs)

**Use Cases**:
- When you are not sure which agent to use
- Tasks that require both coding and file/log operations
- End‑to‑end troubleshooting flows (e.g., “analyze logs and then generate a fix script”)

### 2. Coder Agent

Specializes in programming and software development tasks, with a focus on **Python** and **PowerShell** scripting for Windows environments.

**Features**:
- Write and analyze code (Python, PowerShell, plus other languages via tools)
- Debug and test code, including running unit tests
- Code optimization and refactoring
- Execute scripts/snippets via the execution tool
- Can call File Manager and Log Analyzer via the collaboration tool

**Use Cases**:
- “Write a PowerShell script to collect Windows event logs for a given EventID”
- “Create a Python script to parse and summarize IIS logs”
- “Analyze performance issues in this function and optimize it”
- “Generate unit tests for this module”

### 3. File Manager Agent

Specializes in file system and document management tasks, operating inside the configured working directory.

**Features**:
- Search files and directories (with filtering and size/depth info)
- Read and write file contents
- Create and organize project directory structures
- Document analysis (text, docx, PDF)
- Can call Coder and Log Analyzer via the collaboration tool

**Use Cases**:
- “Find all `.log` and `.txt` files under the working directory”
- “Read and summarize this PDF / DOCX”
- “Create a standard project directory structure for a Python project”

### 4. Log Analyzer Agent

Specialized in Windows OS log analysis and troubleshooting.

**Features**:
- Search log files with regex and filters (size, path, etc.)
- Extract and summarize error and warning patterns
- Root‑cause analysis over log files (timeline, common error types)
- Windows Event Log (EVT/EVTX) querying (requires `pywin32` on Windows)
- Can call Coder to generate fix scripts (Python/PowerShell) or File Manager to explore more files

**Use Cases**:
- “Analyze `System` and `Application` logs to find causes of random reboots”
- “Summarize the root cause of these IIS errors from logs in the working directory”
- “Search Windows Event Logs for EventID 1000 errors and summarize them”

## 🛠️ Available Tools

### Local Tools

The system includes a comprehensive set of local tools for various tasks:

#### File Operations
- **file_read_tool**: Read text files with encoding support
- **list_filename_tool**: List files and directories with filtering options
- **write_txt_tool**: Write content to text files with timestamp support

#### Document Processing
- **doc_read_tool**: Read Microsoft Word documents (.docx) including text and table content
- **pdf_read_tool**: Extract text from PDF files using PyPDF2 or pdfplumber

#### Code Development
- **code_analyzer_tool**: Analyze code for quality issues, bugs, security vulnerabilities, and performance problems
- **code_execution_tool**: Execute code snippets in multiple languages (Python, PowerShell, JavaScript, Java, C/C++, Bash)
- **code_search_tool**: Search for code patterns, functions, classes, or text within code files
- **code_test_tool**: Run tests using various frameworks (pytest, unittest, jest, mocha, junit)
- **code_writer_tool**: Write or modify code files with proper formatting and documentation
- **run_unit_tests_tool**: Execute unit tests using uv command with pytest

#### System Operations
- **run_command_tool**: Execute shell commands with timeout and working directory options
- **agent_collaboration_tool**: Allow sub‑agents to call each other (`coder`, `file_manager`, `log_analyzer`) for cross‑domain tasks

#### Log Analysis
- **log_analysis_tool**: Search and analyze log files for errors and patterns
- **Windows Event Log helper**: (within `log_analysis_tool`) query Windows Event Logs by log name, level, event ID, etc.

### MCP Integrations

- **Desktop Commander**: File system operations and desktop automation
- **Python Sandbox**: Secure code execution environment
- **DuckDuckGo Search**: Web search capabilities
- **GitHub MCP**: Repository management and operations

## 📖 Usage Examples

Try these prompts to explore the agents' capabilities:

### Orchestrator Agent Examples
- "Help me analyze this project's code structure and generate documentation"
- "Create a Python project including setup files and basic functionality"
- "Search for all test files in this project and run tests"

### Coder Agent Examples
- "Write a Python script to process CSV data"
- "Optimize the performance of this code"
- "Write unit tests for this function"

### File Manager Agent Examples
- "Find all configuration files in the project"
- "Read requirements.txt and analyze dependencies"
- "Create a standard project directory structure"

## 🏗️ Architecture

The agent system uses a state graph architecture with the following components:

1. **Orchestration Layer**: Analyzes user requests and selects appropriate agents
2. **Agent Layer**: Specialized agents handle specific types of tasks
3. **Tool Layer**: Local tools and MCP server integrations
4. **Checkpoint System**: Saves conversation state for continuity

## 🔧 Development

### Building MCP Docker Images

```bash
# Build the Deno MCP Docker image
docker build -t deno-docker:latest -f ./mcps/deno/Dockerfile .
```

### Running the GitHub MCP Server

```bash
docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=your_token ghcr.io/github/github-mcp-server
```

### Inspecting the Database

The project uses SQLite to store checkpoints:

```bash
sqlite3 checkpoints.db

# List all tables
.tables

# Show table schema
.schema checkpoints

# Export query results
.mode csv
.output results.csv
.headers on
SELECT * FROM checkpoints;
.output stdout
```

## 🐛 Troubleshooting

### Common Issues

1. **uv command fails**
   - Ensure you've run `uv init` and `uv add -r requirements.txt`
   - Verify the virtual environment is activated

2. **Ollama connection issues**
   - Check that Ollama is running: `ollama list`
   - Verify the model specified in `.env` is downloaded

3. **Docker errors**
   - Confirm Docker Desktop is running
   - Check Docker permissions

4. **Python version mismatch**
   - Use the Python version your virtual environment was created with
   - Recreate the virtual environment if needed

5. **Model response errors**
   - Check your API keys in the `.env` file
   - Verify the model name matches what's available

### Debug Mode

For detailed debugging, you can modify the agent initialization in `agent.py` to enable verbose logging.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔒 Security Considerations

- The agent reads files but does not execute arbitrary shell commands
- Review tools before trusting them with sensitive directories
- API keys should be stored securely in environment variables
- MCP servers run in isolated environments when possible

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on the repository.

---