from typing import Annotated, Sequence
from dotenv import load_dotenv
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from tools.run_unit_tests_tool import run_unit_tests
from tools.doc_read_tool import DocReadTool
from tools.list_filename_tool import ListFilesTool
from tools.pdf_read_tool import PdfReadTool
from tools.write_txt_tool import WriteMemoryTool, ReadMemoryTool, ListMemoriesTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# import sqlite3
# import aiosqlite


class AgentState(BaseModel):
    """
    Persistent agent state tracked across the graph.
    - messages: complete chat history (system + user + assistant + tool messages)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class Agent:
    def __init__(self, config_path: str = "config.json"):
        self._initialized = False
        # Load environment
        load_dotenv()
        
        # Rich console for UI (set before _load_config in case it needs to print)
        self.console = Console()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Agent registry for inter-agent collaboration
        self.agent_registry = None
        
        # Get model type from config
        self.model_type = self.config.get("model", {}).get("type", "ollama")
        
        # Initialize model based on config
        if self.model_type == "ollama":
            ollama_config = self.config.get("model", {}).get("ollama", {})
            file_manager_config = self.config.get("model", {}).get("file_manager", {})
            
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            ollama_model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = file_manager_config.get("temperature", 0.3)
            
            self.model = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=temperature,
                max_tokens=ollama_config.get("max_tokens", 4096),
            )
            self.console.print(f"[green]Using Ollama model: {ollama_model} at {ollama_base_url}[/green]")
        else:  # cloud API
            cloud_config = self.config.get("model", {}).get("cloud", {})
            file_manager_config = self.config.get("model", {}).get("file_manager", {})
            
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(file_manager_config.get("model_env", "OPENAI_MODEL_FILE"), cloud_config.get("model", "gpt-4"))
            temperature = file_manager_config.get("temperature", 0.3)
            
            if not api_key:
                self.console.print("[red]Error: OPENAI_API_KEY environment variable is not set![/red]")
                self.console.print("[yellow]Please set the OPENAI_API_KEY in your .env file.[/yellow]")
                exit(1)
            
            self.model = ChatOpenAI(
                base_url=api_base,
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=cloud_config.get("max_tokens", 4096),
            )
            self.console.print(f"[green]Using Cloud API model: {model_name} at {api_base}[/green]")

        # Build workflow graph
        self.workflow = StateGraph(AgentState)

        # Register nodes
        self.workflow.add_node("user_input", self.user_input)
        self.workflow.add_node("model_response", self.model_response)
        self.workflow.add_node("tool_use", self.tool_use)

        # Edges: start at user_input
        self.workflow.set_entry_point("user_input")
        #self.workflow.add_edge("user_input", "model_response")
        self.workflow.add_edge("tool_use", "model_response")

        self.workflow.add_conditional_edges("user_input", self.check_exit,
            {
                "exit": END,
                "user_input": "model_response",
            }
        )

        # Conditional: model_response -> tool_use OR -> user_input
        self.workflow.add_conditional_edges(
            "model_response",
            self.check_tool_use,
            {
                "tool_use": "tool_use",
                "user_input": "user_input",
            },
        )

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.console.print(f"[yellow]Warning: Config file '{config_path}' not found. Using defaults.[/yellow]")
            return {}
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Error: Invalid JSON in config file: {e}[/red]")
            return {}
    
    def set_agent_registry(self, agent_registry: dict):
        """Set the agent registry for inter-agent collaboration."""
        self.agent_registry = agent_registry
    
    async def ask_fresh_start(self):
        """
        Ask user if they want to start fresh with a new conversation.
        If yes, delete memory from checkpoints.db; otherwise continue previous conversation.
        """
        self.console.print(
            Panel.fit(
                Markdown("Welcome to my agent!"),
                title="[bold yellow]Start up[/bold yellow]",
                border_style="blue",
            )
        )

        while True:
            self.console.print("[bold cyan]Do you want to start a fresh conversation?[/bold cyan]")
            self.console.print("[dim]Enter 'y' or 'yes' to start fresh(will clear previous memory)[/dim]")
            self.console.print("[dim]Enter 'n' or 'no' to continue previous conversation[/dim]")
            
            user_input = self.console.input("\n[bold cyan]Enter your choice:[/bold cyan] ").strip().lower()

            if user_input in ['y','yes','new']:
                # Delete memory from checkpoints.db
                self.console.print("[bold yellow]Deleting previous conversation memory...[/bold yellow]")
                # Delete checkpoints.db
                db_path = os.path.join(os.getcwd(), "checkpoints.db")
                try:
                    if os.path.exists(db_path):
                        os.remove(db_path)
                        self.console.print(
                            Panel.fit(
                                Markdown("[bold green]Successfully deleted previous memory![/bold green]"),
                                title="[bold green]Success[/bold green]",
                                border_style="green",
                            )
                        )
                    else:
                        self.console.print(
                            Panel.fit(
                                Markdown("[bold yellow]No previous memory found to delete.[/bold yellow]"),
                                title="[bold yellow]Warning[/bold yellow]",
                                border_style="yellow",
                            )
                        )
                    
                except Exception as e:
                    self.console.print(f"[bold red]Error deleting memory: {e}[/bold red]")
                    self.console.print("[bold yellow]Continue previous conversation.[/bold yellow]")
                return True

            elif user_input in ['n','no','continue']:
                self.console.print(
                    Panel.fit(
                        Markdown("[bold green]Continue previous conversation.[/bold green]"),
                        title="[bold green]Warning[/bold green]",
                        border_style="green",
                    )
                )
                return False
            else:
                self.console.print("[bold red]Invalid input. Please enter 'y' or 'n'.[/bold red]")

    async def initialize(self):
        """Async initialization - load tools and other async resources"""
        if self._initialized:
            return self

        print("🔄 Initializing agent...")

        # Tools
        local_tools = [
            run_unit_tests,
            DocReadTool(),
            ListFilesTool(),
            PdfReadTool(),
            WriteMemoryTool(),
            ReadMemoryTool(),
            ListMemoriesTool()
        ]
        
        # Add agent collaboration tool if registry is available
        if self.agent_registry:
            from tools.agent_collaboration_tool import call_other_agent
            from langchain_core.tools import tool
            
            # Create a bound version of call_other_agent with registry
            def call_agent_with_registry(agent_name: str, request: str) -> str:
                """Call another agent (coder or log_analyzer) to help with a task.
                
                Args:
                    agent_name: Name of the agent to call ("coder" or "log_analyzer")
                    request: The request or question to send to the other agent
                
                Returns:
                    Response from the called agent
                """
                return call_other_agent.invoke({
                    "agent_name": agent_name,
                    "request": request,
                    "agent_registry": self.agent_registry
                })
            
            # Create a tool wrapper
            collaboration_tool = tool(call_agent_with_registry)
            collaboration_tool.name = "call_other_agent"
            local_tools.append(collaboration_tool)

        # Set up MCP client
        mcp_tools = await self.get_mcp_tools()
        self.tools = local_tools + mcp_tools
        print(
            f"✅ Loaded {len(self.tools)} total tools (Local: {len(local_tools)} + MCP: {len(mcp_tools)})"
        )
        self._initialized = True

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)

        # Compile graph
        #async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        #    self.agent = self.workflow.compile(checkpointer=memory)
        # Compile graph: enter AsyncSqliteSaver once and keep it open for agent lifetime
        # (prevents re-opening/closing aiosqlite threads repeatedly)
        agent_config = self.config.get("agents", {}).get("file_manager", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)

        # Optional: print a greeting panel
        self.console.print(
            Panel.fit(
                Markdown("**LangGraph Agent**"),
                title="[bold green]Ready[/bold green]",
                border_style="green",
            )
        )
        return self

    async def run(self):
        """
        Main entry point: invoke the agent with a default message.
        The workflow internally handles the conversation loop and exit logic.
        """
        agent_config = self.config.get("agents", {}).get("file_manager", {})
        config = {
            "configurable": {"thread_id": agent_config.get("thread_id", "1")},
            "recursion_limit": self.config.get("ui", {}).get("recursion_limit", 100)
        }
        initial_state = {"messages": AIMessage(content="What can I do for you?")}

        try:
            await self.agent.ainvoke(initial_state, config=config)
            self.console.print(
                Panel.fit(
                    Markdown("Conversation ended!"),
                    title="Bye",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            self.console.print(
                Panel.fit(
                    Markdown("Conversation interrupted by user!"),
                    title="Warning",
                    border_style="yellow",
                )
            )
        except Exception as e:
            self.console.print(f"Error: {e}")
        finally:
            await self.close_checkpointer()

    async def close_checkpointer(self):
        """Close the async checkpointer context if opened."""
        if hasattr(self, "_checkpointer_ctx"):
            await self._checkpointer_ctx.__aexit__(None, None, None)

    async def get_mcp_tools(self):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        import shutil
        if shutil.which("docker"):
            try:
                mcp_client = MultiServerMCPClient(
                    {
                        "Run_Python_MCP": {
                            "command": "docker",
                            "args": [
                                "run",
                                "-i",
                                "--rm",
                                "deno-docker:latest",  # image name
                                "deno",  # the command inside container
                                "run",
                                "-N",
                                "-R=node_modules",
                                "-W=node_modules",
                                "--node-modules-dir=auto",
                                "jsr:@pydantic/mcp-run-python",
                                "stdio",
                            ],
                            "transport": "stdio",
                        },
                        "duckduckgo_MCP": {
                            "command": "docker",
                            "args": ["run", "-i", "--rm", "mcp/duckduckgo"],
                            "transport": "stdio",
                        },
                        "desktop_commander_in_docker_MCP": {
                            "command": "docker",
                            "args": [
                                "run",
                                "-i",
                                "--rm",
                                "-v",
                                "/Users/lorreatlan/Documents/MyPlayDocuments:/mnt/documents",
                                "mcp/desktop-commander:latest",
                            ],
                            "transport": "stdio",
                        },
                        "Github_MCP": {
                            "command": "docker",
                            "args": [
                                "run",
                                "-i",
                                "--rm",
                                "-e",
                                f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_PERSONAL_ACCESS_TOKEN}",
                                "-e",
                                "GITHUB_READ-ONLY=1",
                            ],
                            "transport": "stdio",
                        },
                        "github_code_search_MCP": {
                            "command": "docker",
                            "args": [
                                "run",
                                "-i",
                                "--rm",
                                "-e",
                                f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_PERSONAL_ACCESS_TOKEN}",
                                "-e",
                                "GITHUB_READ-ONLY=1",
                                "ghcr.io/github/github-mcp-server",
                            ],
                            "transport": "stdio",
                        },
                    }
                )
                mcp_tools = await mcp_client.get_tools()
                for tb in mcp_tools:
                    print(f"MCP 🔧 {tb.name}")
                return mcp_tools
            except Exception as e:
                self.console.print(f"Error importing MCP tools: {e}")
                self.console.print(f"Using local tools instead.")
                return []
        else:
            self.console.print(f"Docker is not installed. Using local tools instead.")
            return []

    # Node: user_input
    def user_input(self, state: AgentState) -> AgentState:
        """
        Ask user for input and append HumanMessage to state.
        Handles empty input by asking again.
        Supports exit commands to quit the conversation.
        """
        while True:
            self.console.print("[bold cyan]User Input[/bold cyan]: ")
            user_input = self.console.input("> ").strip()

            # check empty input
            if not user_input:
                self.console.print("Empty input is not allowed. Please try again.")
                self.console.print("Tip: Type 'exit', 'bye' or 'quit' to end the conversation.\n")
                continue
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                self.console.print(
                    Panel.fit(
                        Markdown("Thank you for using, good bye!"),
                        title="Conversation ended",
                        border_style="cyan",
                    )
                )
                return {"messages": [HumanMessage(content="__EXIT__")]}
            return {"messages": [HumanMessage(content=user_input)]}

    # Node: model_response
    def model_response(self, state: AgentState) -> AgentState:
        """
        Call the LLM (with tools bound). Print assistant content and any tool_call previews.
        Decide routing via check_tool_use.
        """
        #system_text = """You are a specialised agent for maintaining and developing codebases.
        #    ## Development Guidelines:

        #    1. **Test Failures:**
        #    - When tests fail, fix the implementation first, not the tests.
        #    - Tests represent expected behavior; implementation should conform to tests
        #    - Only modify tests if they clearly don't match specifications

        #    2. **Code Changes:**
        #    - Make the smallest possible changes to fix issues
        #    - Focus on fixing the specific problem rather than rewriting large portions
        #    - Add unit tests for all new functionality before implementing it

        #    3. **Best Practices:**
        #    - Keep functions small with a single responsibility
        #    - Implement proper error handling with appropriate exceptions
        #    - Be mindful of configuration dependencies in tests

        #    Ask for clarification when needed. Remember to examine test failure messages carefully to understand the root cause before making any changes."""
        
        system_text = """
        You are a helpful File Assistant Agent. Your job is to help users search for information, 
        answer questions, and assist with simple tasks related to files on their computer.
        
        ## Capabilities:

        - Search for files and directories using glob patterns
        - Read file contents
        - Write file contents
        - Create new files and directories
        - Answer questions about file contents (if provided)
        - Use provided tools to search data based on question. Tips: use the write_txt_tool to mark down 
          the related content and answer question based on the record you just marked down
        - Respond in a concise and friendly manner

        ## ReAct Framework:

        You should follow the ReAct (Reasoning and Acting) framework:
        1. **Think**: Analyze the user's request and plan your approach
        2. **Act**: Use appropriate tools to gather information
        3. **Observe**: Review the tool results
        4. **Think**: Analyze the observations and determine next steps
        5. **Act/Observe/Think**: Repeat until you have enough information
        6. **Answer**: Provide a comprehensive answer based on your findings

        ## Instructions:

        - Always follow the given instructions and capabilities
        - If a user asks for something that is not within the capabilities, politely inform them that you cannot help with that
        - If a user asks for something that is not clear, ask them for more information
        - For file operations, always ask for confirmation before performing the action
        - Keep responses short and easy to understand
        - Never share, modify, or delete any files or directories without explicit user permission

        ## Response Format:

        - Use numbered or bulleted lists for multiple items
        - Use bold text for emphasis
        - Use code blocks for any code snippets or file paths
        - Use simple language
        - Always provide a clear and concise response

        ## Examples:

        - "I found the following files: file1.txt, file2.txt. Would you like to read the content inside?"
        - "I can help you search for documents, please provide a question."

        ## Inter-Agent Collaboration:
        
        If you need capabilities from other agents, use the call_other_agent tool:
        - Use "coder" agent for code generation, script creation, code analysis
        - Use "log_analyzer" agent for log analysis, troubleshooting, Windows Event Log access
        
        Example: If you find a script file and need to understand or modify it, call the coder agent.
        
        ## Remember:
        Always think step-by-step, use tools to gather information, observe results, and then provide answers.
        Your goal is to make file management easy and safe for end users. Always prioritize clarity, safety and helpfulness in your responses.
        Don't hesitate to collaborate with other agents when their expertise is needed.
        """
        # Compose messages: include prior state
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(content=f"Working directory: {os.path.expandvars(self.config.get('working_directory', os.getcwd()))}"),
        ] + state.messages

        # Invoke model with thinking indicator
        from rich.spinner import Spinner
        from rich.live import Live
        with Live(Spinner("point", "[bold magenta]Thinking...[/bold magenta]"), refresh_per_second=10) as live:
            response = self.model_with_tools.invoke(messages)
            live.stop()
        if isinstance(response.content, list):
            for item in response.content:
                if item["type"] == "text":
                    text = item.get("text", "")
                    if text:
                        self.console.print(
                            Panel.fit(
                                Markdown(text),
                                title="[magenta]Assistant[/magenta]",
                                border_style="magenta",
                            )
                        )
                elif item["type"] == "tool_use":
                    self.console.print(
                        Panel.fit(
                            Markdown(
                                f"{item['name']} with args {item.get('args',None)}"
                            ),
                            title="Tool Use",
                        )
                    )
        else:
            self.console.print(
                Panel.fit(
                    Markdown(response.content),
                    title="[magenta]Assistant[/magenta]",
                )
            )

        return {"messages": [response]}

    # Conditional router
    def check_tool_use(self, state: AgentState) -> str:
        """
        Route based on the last message:
        - If tool_calls exist -> 'tool_use'
        - If __EXIT__ signal -> 'END' (terminate workflow)
        - Otherwise -> 'user_input' (continue conversation)
        """
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        return "user_input"

    # Node: tool_use
    async def tool_use(self, state: AgentState) -> AgentState:
        """
        Execute tool calls from the last assistant message and return ToolMessage(s),
        preserving tool_call_id so the model can reconcile results when we go back to model_response.
        """
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import AIMessage

        response = []
        tools_by_name = {t.name: t for t in self.tools}

        # Check if the last message is an AIMessage with tool calls
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            print(f"❌ Error: Last message is not an AIMessage with tool_calls. Message type: {type(last_message)}")
            return {"messages": [HumanMessage(content="Error: No valid tool calls found in the last message.")]}
        
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"🔧 Invoking tool '{tool_name}' with args {tool_args}")
            tool = tools_by_name.get(tool_name)
            print(f"🛠️ Found tool: {tool}")
            tool_node = ToolNode([tool])

            # response = interrupt(
            #     {
            #         "action": "review_tool_call",
            #         "tool_name": tool_name,
            #         "tool_input": state["messages"][-1].content,
            #         "message": "Approve this tool call?",
            #     }
            # )
            # # Handle the response after the interrupt (e.g., resume or modify)
            # if response == "approved":
            try:
                from rich.spinner import Spinner
                from rich.live import Live
                with Live(Spinner("point", f"[bold magenta]Executing {tool_name}...[/bold magenta]"), refresh_per_second=10) as live:
                    tool_result = await tool_node.ainvoke(state)
                    live.stop()

                print(f"🛠️ Tool Result: {tool_result}")
                response.append(tool_result["messages"][0])
                self.console.print(
                    Panel.fit(
                        Syntax(
                            "\n" + tool_result["messages"][0].content + "\n", "text"
                        ),
                        title="Tool Result",
                    )
                )
            except Exception as e:
                response.append(
                    ToolMessage(
                        content=f"ERROR: Exception during tool '{tool_name}' execution: {e}",
                        tool_call_id=tc["id"],
                    )
                )
                self.console.print(
                    Panel.fit(
                        Markdown(
                            f"**ERROR**: Exception during tool '{tool_name}' execution: {e}"
                        ),
                        title="Tool Error",
                        border_style="red",
                    )
                )
            # else:
            #     # Handle rejection or modification
            #     pass
        return {"messages": response}

    def check_exit(self, state: AgentState) -> str:
        """
        Check if the last user message is an exit command.
        """
        last_message = state.messages[-1]
        if hasattr(last_message, "content") and last_message.content == "__EXIT__":
            return "exit"
        return "user_input"

    def print_mermaid_workflow(self):
        """
        Utility: print Mermaid diagram to visualize the graph edges.
        """
        try:
            mermaid = self.agent.get_graph().draw_mermaid_png(
                output_file_path="langgraph_workflow.png",
                max_retries=5,
                retry_delay=2,
            )
        except Exception as e:
            print(f"Error generating mermaid PNG: {e}")
            mermaid = self.agent.get_graph().draw_mermaid()
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            print(self.agent.get_graph().draw_ascii())
