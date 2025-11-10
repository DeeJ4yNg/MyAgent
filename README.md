# CLI Agent

A powerful, interactive coding assistant built with LangGraph, LangChain, and Ollama/OpenAI models. This project provides a terminal-based AI agent with local utility tools and support for remote MCP (Model Context Protocol) servers.

![LangGraph Workflow](langgraph_workflow.png)

## 🌟 Features

- **Interactive Agent**: State-driven workflow that processes user input, generates model responses, and executes tools
- **Multiple Model Support**: Choose between local Ollama models or cloud-based OpenAI-compatible APIs
- **Rich Terminal UI**: Beautiful console interface with Markdown rendering and syntax highlighting
- **Local Tools**: File operations, document reading, PDF processing, and unit test execution
- **MCP Integration**: Support for remote MCP servers including Desktop Commander, Python sandbox, DuckDuckGo search, and GitHub
- **Persistent Memory**: SQLite-based checkpoint system for conversation continuity
- **Workflow Visualization**: Automatic generation of Mermaid diagrams for the agent's workflow

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
   cd claude-code-clone
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

### Running the Agent

```bash
# Using uv
uv run main.py

# Or with Python directly (after activating venv)
python main.py
```

## 🛠️ Available Tools

### Local Tools

- **File Operations**: Read, write, and list files
- **Document Processing**: Read Word documents and PDFs
- **Unit Testing**: Run pytest and return results
- **Memory Management**: Store and retrieve conversation memories

### MCP Integrations

- **Desktop Commander**: File system operations and desktop automation
- **Python Sandbox**: Secure code execution environment
- **DuckDuckGo Search**: Web search capabilities
- **GitHub MCP**: Repository management and operations

## 📖 Usage Examples

Try these prompts to explore the agent's capabilities:

- "Summarize the recent articles from https://simonwillison.net/"
- "Use python_run_code tool to run ascii_art_generator.py"
- "Show me the content of main.py"
- "What tools do you have?"
- "Read requirements.txt"
- "Run unit tests for this project"

## 🏗️ Architecture

The agent is built using a state graph architecture with the following components:

1. **User Input Node**: Captures and processes user commands
2. **Model Response Node**: Generates AI responses using the selected LLM
3. **Tool Use Node**: Executes tools based on model decisions
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

**Note**: This project is a demonstration of LangGraph capabilities and is not affiliated with Anthropic's Claude.