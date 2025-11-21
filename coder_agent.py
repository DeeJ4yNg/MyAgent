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
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from tools.token_tracker import tracker


class CoderAgentState(BaseModel):
    """
    Persistent agent state tracked across the graph.
    - messages: complete chat history (system + user + assistant + tool messages)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class CoderAgent:
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
            coder_config = self.config.get("model", {}).get("coder", {})
            
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            ollama_model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = coder_config.get("temperature", 0.1)
            
            self.model = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=temperature,
                max_tokens=ollama_config.get("max_tokens", 4096),
            )
            self.console.print(f"[green]Using Ollama model: {ollama_model} at {ollama_base_url}[/green]")
        else:  # cloud API
            cloud_config = self.config.get("model", {}).get("cloud", {})
            coder_config = self.config.get("model", {}).get("coder", {})
            
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(coder_config.get("model_env", "OPENAI_MODEL_CODER"), cloud_config.get("model", "gpt-4"))
            temperature = coder_config.get("temperature", 0.1)
            
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
        self.workflow = StateGraph(CoderAgentState)

        # Register nodes
        self.workflow.add_node("user_input", self.user_input)
        self.workflow.add_node("model_response", self.model_response)
        self.workflow.add_node("tool_use", self.tool_use)

        # Edges: start at user_input
        self.workflow.set_entry_point("user_input")
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
                Markdown("Welcome to the Coder Agent!"),
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
                db_path = os.path.join(os.getcwd(), "coder_checkpoints.db")
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

        print("🔄 Initializing Coder Agent...")

        # Import code-specific tools
        from tools.code_analyzer_tool import analyze_code
        from tools.code_writer_tool import write_code
        from tools.code_search_tool import search_code
        from tools.code_test_tool import run_code_tests
        from tools.code_execution_tool import execute_code
        from tools.agent_collaboration_tool import call_other_agent

        # Tools
        local_tools = [
            analyze_code,
            write_code,
            search_code,
            run_code_tests,
            execute_code
        ]
        
        # Add agent collaboration tool
        async def call_agent_with_registry(agent_name: str, request: str) -> str:
            """Call another agent (file_manager or log_analyzer) to help with a task."""
            return await call_other_agent.ainvoke({
                "agent_name": agent_name,
                "request": request,
                "agent_registry": self.agent_registry
            })
        from langchain_core.tools import tool
        collaboration_tool = tool(call_agent_with_registry)
        collaboration_tool.name = "call_other_agent"
        local_tools.append(collaboration_tool)

        # Set up MCP client (optional)
        mcp_tools = await self.get_mcp_tools()
        self.tools = local_tools + mcp_tools
        print(
            f"✅ Loaded {len(self.tools)} total tools (Local: {len(local_tools)} + MCP: {len(mcp_tools)})"
        )
        self._initialized = True

        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)

        # Compile graph
        agent_config = self.config.get("agents", {}).get("coder", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "coder_checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)

        # Optional: print a greeting panel
        self.console.print(
            Panel.fit(
                Markdown("**Coder Agent** - Your AI coding assistant"),
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
        agent_config = self.config.get("agents", {}).get("coder", {})
        config = {
            "configurable": {"thread_id": agent_config.get("thread_id", "coder-1")},
            "recursion_limit": self.config.get("ui", {}).get("recursion_limit", 100)
        }
        initial_state = {"messages": AIMessage(content="What coding task can I help you with today?")}

        try:
            await self.agent.ainvoke(initial_state, config=config)
            self.console.print(
                Panel.fit(
                    Markdown("Coding session ended!"),
                    title="Bye",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            self.console.print(
                Panel.fit(
                    Markdown("Coding session interrupted by user!"),
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
        # Optional: Add MCP tools if needed
        # For now, returning empty list
        return []

    # Node: user_input
    def user_input(self, state: CoderAgentState) -> CoderAgentState:
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
                        Markdown("Thank you for using Coder Agent, good bye!"),
                        title="Conversation ended",
                        border_style="cyan",
                    )
                )
                return {"messages": [HumanMessage(content="__EXIT__")]}
            return {"messages": [HumanMessage(content=user_input)]}

    # Node: model_response
    def model_response(self, state: CoderAgentState) -> CoderAgentState:
        """
        Call the LLM (with tools bound). Print assistant content and any tool_call previews.
        Decide routing via check_tool_use.
        """
        system_text = """
        You are a specialized Coder Agent designed to help with software development tasks, 
        with particular expertise in Python and PowerShell scripting for Windows OS support.
        Your expertise includes code analysis, writing, debugging, testing, and optimization.
        
        ## Capabilities:
        
        - Analyze existing code for bugs, performance issues, and best practices
        - Write new code in various programming languages (especially Python and PowerShell)
        - Create PowerShell scripts for Windows administration and automation
        - Create Python scripts for system management and automation
        - Search through codebases to find specific functions or patterns
        - Run tests and analyze test results
        - Execute code snippets and show results
        - Refactor code for better readability and performance
        - Suggest improvements to existing implementations
        
        ## ReAct Framework:
        
        You should follow the ReAct (Reasoning and Acting) framework:
        1. **Think**: Analyze the user's request and plan your approach
        2. **Act**: Use appropriate tools to gather information or write code
        3. **Observe**: Review the tool results or code execution output
        4. **Think**: Analyze the observations and determine next steps
        5. **Act/Observe/Think**: Repeat until you have a complete solution
        6. **Answer**: Provide a comprehensive solution with explanations
        
        ## Python & PowerShell Focus:
        
        - **Python**: Emphasize clean, readable code with proper error handling
        - **PowerShell**: Follow PowerShell best practices, use proper cmdlets, handle errors with try-catch
        - **Windows Integration**: Consider Windows-specific APIs and PowerShell modules
        - **Scripting**: Create reusable, parameterized scripts for automation
        
        ## Guidelines:
        
        1. **Code Quality**: Always write clean, well-documented code following best practices
        2. **Testing**: Encourage testing and provide test cases when appropriate
        3. **Security**: Consider security implications when suggesting code changes
        4. **Performance**: Optimize for performance when relevant
        5. **Clarity**: Explain complex concepts in simple terms
        6. **Windows Context**: Consider Windows OS specifics when writing scripts
        
        ## Response Format:
        
        - Use code blocks for all code snippets with proper language tags (python, powershell, etc.)
        - Explain your reasoning before providing code solutions
        - Highlight potential issues or considerations
        - Suggest follow-up improvements when relevant
        - For PowerShell, include proper error handling and parameter validation
        
        ## Inter-Agent Collaboration:
        
        If you need capabilities from other agents, use the call_other_agent tool:
        - Use "file_manager" agent for file operations, searching files, reading documents
        - Use "log_analyzer" agent for log analysis, troubleshooting, Windows Event Log access
        
        Example: If you need to read a configuration file before writing code, call the file_manager agent.
        
        ## Remember:
        Always think step-by-step, use tools to gather information, observe results, and then provide solutions.
        Create a plan before writing any code.
        Your goal is to help users become better developers by providing high-quality code solutions and explanations.
        Always prioritize code quality, security, and maintainability in your responses.
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
                                title="[magenta]Coder Assistant[/magenta]",
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
                    title="[magenta]Coder Assistant[/magenta]",
                )
            )

        try:
            prompt_text = ""
            for m in messages:
                c = m.content
                if isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_text += item.get("text", "")
                elif isinstance(c, str):
                    prompt_text += c
                else:
                    prompt_text += str(c)
            if isinstance(response.content, list):
                completion_text = ""
                for item in response.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        completion_text += item.get("text", "")
            else:
                completion_text = response.content if isinstance(response.content, str) else str(response.content)
            tracker.record("coder", tracker.estimate_tokens(prompt_text), tracker.estimate_tokens(completion_text))
        except Exception:
            pass
        return {"messages": [response]}

    # Conditional router
    def check_tool_use(self, state: CoderAgentState) -> str:
        """
        Route based on the last message:
        - If tool_calls exist -> 'tool_use'
        - If __EXIT__ signal -> 'END' (terminate workflow)
        - Otherwise -> 'user_input' (continue conversation)
        """
        last_message = state.messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        for msg in reversed(state.messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return "tool_use"
        return "user_input"

    # Node: tool_use
    async def tool_use(self, state: CoderAgentState) -> CoderAgentState:
        """
        Execute tool calls from the last assistant message and return ToolMessage(s),
        preserving tool_call_id so the model can reconcile results when we go back to model_response.
        """
        from langgraph.prebuilt import ToolNode
        from langchain_core.messages import AIMessage

        response = []
        tools_by_name = {t.name: t for t in self.tools}

        ai_msg = None
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            ai_msg = last_message
        else:
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                    ai_msg = msg
                    break
        if not ai_msg:
            return {"messages": []}

        for tc in ai_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"🔧 Invoking tool '{tool_name}' with args {tool_args}")
            tool = tools_by_name.get(tool_name)
            print(f"🛠️ Found tool: {tool}")
            tool_node = ToolNode([tool])

            try:
                from rich.spinner import Spinner
                from rich.live import Live
                with Live(Spinner("point", f"[bold magenta]Executing {tool_name}...[/bold magenta]"), refresh_per_second=10) as live:
                    tool_result = await tool_node.ainvoke({"messages": [ai_msg]})
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
        return {"messages": response}

    def check_exit(self, state: CoderAgentState) -> str:
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
                output_file_path="coder_workflow.png",
                max_retries=5,
                retry_delay=2,
            )
        except Exception as e:
            print(f"Error generating mermaid PNG: {e}")
            mermaid = self.agent.get_graph().draw_mermaid()
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Coder Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            print(self.agent.get_graph().draw_ascii())