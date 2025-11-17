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


class LogAnalyzerAgentState(BaseModel):
    """
    Persistent agent state tracked across the graph.
    - messages: complete chat history (system + user + assistant + tool messages)
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class LogAnalyzerAgent:
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
            log_config = self.config.get("model", {}).get("log_analyzer", {})
            
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            ollama_model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = log_config.get("temperature", 0.2)
            
            self.model = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=temperature,
                max_tokens=ollama_config.get("max_tokens", 4096),
            )
            self.console.print(f"[green]Using Ollama model: {ollama_model} at {ollama_base_url}[/green]")
        else:  # cloud API
            cloud_config = self.config.get("model", {}).get("cloud", {})
            log_config = self.config.get("model", {}).get("log_analyzer", {})
            
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(log_config.get("model_env", "OPENAI_MODEL_LOG"), cloud_config.get("model", "gpt-4"))
            temperature = log_config.get("temperature", 0.2)
            
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
        self.workflow = StateGraph(LogAnalyzerAgentState)

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

    async def initialize(self):
        """Async initialization - load tools and other async resources"""
        if self._initialized:
            return self

        print("🔄 Initializing Log Analyzer Agent...")
        
        # Import log analysis tools
        from tools.log_analysis_tool import (
            search_log_files,
            analyze_log_errors,
            summarize_log_root_cause,
            find_windows_event_logs
        )
        from tools.file_read_tool import FileReadTool
        from tools.list_filename_tool import ListFilesTool

        # Tools
        local_tools = [
            search_log_files,
            analyze_log_errors,
            summarize_log_root_cause,
            find_windows_event_logs,
            FileReadTool(),
            ListFilesTool()
        ]
        
        # Add agent collaboration tool if registry is available
        if self.agent_registry:
            from tools.agent_collaboration_tool import call_other_agent
            from langchain_core.tools import tool
            
            # Create a bound version of call_other_agent with registry
            def call_agent_with_registry(agent_name: str, request: str) -> str:
                """Call another agent (coder or file_manager) to help with a task.
                
                Args:
                    agent_name: Name of the agent to call ("coder" or "file_manager")
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
        agent_config = self.config.get("agents", {}).get("log_analyzer", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "log_analyzer_checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)

        # Optional: print a greeting panel
        self.console.print(
            Panel.fit(
                Markdown("**Log Analyzer Agent** - Your AI log analysis assistant"),
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
        agent_config = self.config.get("agents", {}).get("log_analyzer", {})
        config = {
            "configurable": {"thread_id": agent_config.get("thread_id", "log-analyzer-1")},
            "recursion_limit": self.config.get("ui", {}).get("recursion_limit", 100)
        }
        initial_state = {"messages": AIMessage(content="I'm your log analyzer agent. I can help you analyze log files and identify root causes of issues. What logs would you like me to analyze?")}

        try:
            await self.agent.ainvoke(initial_state, config=config)
            self.console.print(
                Panel.fit(
                    Markdown("Log analysis session ended!"),
                    title="Bye",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            self.console.print(
                Panel.fit(
                    Markdown("Log analysis session interrupted by user!"),
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
        return []

    # Node: user_input
    def user_input(self, state: LogAnalyzerAgentState) -> LogAnalyzerAgentState:
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
                        Markdown("Thank you for using Log Analyzer Agent, good bye!"),
                        title="Conversation ended",
                        border_style="cyan",
                    )
                )
                return {"messages": [HumanMessage(content="__EXIT__")]}
            return {"messages": [HumanMessage(content=user_input)]}

    # Node: model_response
    def model_response(self, state: LogAnalyzerAgentState) -> LogAnalyzerAgentState:
        """
        Call the LLM (with tools bound) using ReAct framework.
        Print assistant content and any tool_call previews.
        Decide routing via check_tool_use.
        """
        system_text = """
        You are a specialized Log Analyzer Agent designed to help with Windows OS log analysis and troubleshooting.
        Your expertise includes analyzing log files, identifying error patterns, and determining root causes of issues.
        
        ## Capabilities:
        
        - Search for patterns in log files across directories
        - Analyze log files for errors and warnings
        - Identify root causes of issues from log data
        - Search Windows Event Logs
        - Summarize findings and provide recommendations
        
        ## ReAct Framework:
        
        You should follow the ReAct (Reasoning and Acting) framework:
        1. **Think**: Analyze the user's request and plan your approach
        2. **Act**: Use appropriate tools to gather information
        3. **Observe**: Review the tool results
        4. **Think**: Analyze the observations and determine next steps
        5. **Act/Observe/Think**: Repeat until you have enough information
        6. **Answer**: Provide a comprehensive summary with root cause analysis
        
        ## Guidelines:
        
        1. **Systematic Analysis**: Always start by understanding the issue, then search relevant logs
        2. **Pattern Recognition**: Look for error patterns, timestamps, and correlations
        3. **Root Cause**: Focus on identifying the root cause, not just symptoms
        4. **Windows-Specific**: Be aware of Windows log locations and formats
        5. **Clarity**: Present findings in a clear, structured format
        
        ## Response Format:
        
        - Use structured sections for different aspects of analysis
        - Highlight critical errors and warnings
        - Provide actionable recommendations
        - Include relevant log excerpts with context
        
        ## Inter-Agent Collaboration:
        
        If you need capabilities from other agents, use the call_other_agent tool:
        - Use "coder" agent for creating scripts to fix issues, generating automation code
        - Use "file_manager" agent for searching log files, reading configuration files
        
        Example: If you identify an issue and need to create a fix script, call the coder agent.
        
        ## Remember:
        Always think step-by-step, use tools to gather information, observe results, and then provide a comprehensive analysis.
        Your goal is to help users quickly identify and resolve Windows OS issues through thorough log analysis.
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
        with Live(Spinner("point", "[bold magenta]Analyzing logs...[/bold magenta]"), refresh_per_second=10) as live:
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
                                title="[magenta]Log Analyzer[/magenta]",
                                border_style="magenta",
                            )
                        )
                elif item["type"] == "tool_use":
                    self.console.print(
                        Panel.fit(
                            Markdown(
                                f"**Tool:** {item['name']}\n**Arguments:** {item.get('args', {})}"
                            ),
                            title="Tool Use",
                        )
                    )
        else:
            self.console.print(
                Panel.fit(
                    Markdown(response.content),
                    title="[magenta]Log Analyzer[/magenta]",
                )
            )

        return {"messages": [response]}

    # Conditional router
    def check_tool_use(self, state: LogAnalyzerAgentState) -> str:
        """
        Route based on the last message:
        - If tool_calls exist -> 'tool_use'
        - Otherwise -> 'user_input' (continue conversation)
        """
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        return "user_input"

    # Node: tool_use
    async def tool_use(self, state: LogAnalyzerAgentState) -> LogAnalyzerAgentState:
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
        return {"messages": response}

    def check_exit(self, state: LogAnalyzerAgentState) -> str:
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
                output_file_path="log_analyzer_workflow.png",
                max_retries=5,
                retry_delay=2,
            )
        except Exception as e:
            print(f"Error generating mermaid PNG: {e}")
            mermaid = self.agent.get_graph().draw_mermaid()
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Log Analyzer Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            print(self.agent.get_graph().draw_ascii())

