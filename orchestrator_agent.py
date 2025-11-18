from typing import Annotated, Sequence, Dict, Any, List
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

# Import sub-agents
from coder_agent import CoderAgent
from file_management_agent import Agent
from log_analyzer_agent import LogAnalyzerAgent
from general_assistant_agent import GeneralAssistantAgent


class OrchestratorAgentState(BaseModel):
    """
    Persistent agent state tracked across the graph.
    - messages: complete chat history (system + user + assistant + tool messages)
    - selected_agent: which sub-agent to use (coder, file_manager, or log_analyzer)
    - agent_response: response from the sub-agent
    - active_agent: which agent is currently active (orchestrator, coder, file_manager, or log_analyzer)
    - agent_context: context for the currently active agent
    - coder_memory: conversation history with the coder agent
    - file_manager_memory: conversation history with the file_manager agent
    - log_analyzer_memory: conversation history with the log_analyzer agent
    - plan: execution plan created by orchestrator
    - plan_confirmed: whether the plan has been confirmed by user
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_agent: str = ""
    agent_response: str = ""
    active_agent: str = "orchestrator"  # Default to orchestrator
    agent_context: Dict[str, Any] = {}  # Store context for active agent
    coder_memory: List[BaseMessage] = []  # Store conversation history with coder agent
    file_manager_memory: List[BaseMessage] = []  # Store conversation history with file_manager agent
    log_analyzer_memory: List[BaseMessage] = []  # Store conversation history with log_analyzer agent
    general_memory: List[BaseMessage] = []
    plan: str = ""  # Execution plan
    plan_confirmed: bool = False  # Whether plan is confirmed


class OrchestratorAgent:
    def __init__(self, config_path: str = "config.json"):
        self._initialized = False
        # Load environment
        load_dotenv()
        
        # Rich console for UI (set before _load_config in case it needs to print)
        self.console = Console()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Get model type from config
        self.model_type = self.config.get("model", {}).get("type", "ollama")
        
        # Initialize model based on config
        if self.model_type == "ollama":
            ollama_config = self.config.get("model", {}).get("ollama", {})
            orchestrator_config = self.config.get("model", {}).get("orchestrator", {})
            
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_config.get("base_url", "http://localhost:11434"))
            ollama_model = os.getenv("OLLAMA_MODEL", ollama_config.get("model", "llama3.1:8b"))
            temperature = orchestrator_config.get("temperature", 0.2)
            
            self.model = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=temperature,
                max_tokens=ollama_config.get("max_tokens", 4096),
            )
            self.console.print(f"[green]Using Ollama model: {ollama_model} at {ollama_base_url}[/green]")
        else:  # cloud API
            cloud_config = self.config.get("model", {}).get("cloud", {})
            orchestrator_config = self.config.get("model", {}).get("orchestrator", {})
            
            api_base = os.getenv("OPENAI_API_BASE", cloud_config.get("api_base", "https://api.openai.com/v1"))
            api_key = os.getenv(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
            model_name = os.getenv(orchestrator_config.get("model_env", "OPENAI_MODEL_ORC"), cloud_config.get("model", "gpt-4"))
            temperature = orchestrator_config.get("temperature", 0.2)
            
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
        self.workflow = StateGraph(OrchestratorAgentState)

        # Register nodes
        self.workflow.add_node("user_input", self.user_input)
        self.workflow.add_node("analyze_request", self.analyze_request)
        self.workflow.add_node("create_plan", self.create_plan)
        self.workflow.add_node("wait_for_confirmation", self.wait_for_confirmation)
        self.workflow.add_node("dispatch_to_agent", self.dispatch_to_agent)
        self.workflow.add_node("interact_with_agent", self.interact_with_agent)
        self.workflow.add_node("present_results", self.present_results)

        # Edges: start at user_input
        self.workflow.set_entry_point("user_input")
        
        # Conditional routing based on active_agent
        self.workflow.add_conditional_edges("user_input", self.route_request,
            {
                "analyze": "analyze_request",
                "interact": "interact_with_agent",
                "exit": END
            }
        )
        
        self.workflow.add_conditional_edges("analyze_request", self.check_need_plan,
            {
                "needs_plan": "create_plan",
                "no_plan": "dispatch_to_agent",
            }
        )
        self.workflow.add_conditional_edges("create_plan", self.check_plan_confirmed,
            {
                "confirmed": "dispatch_to_agent",
                "needs_confirmation": "wait_for_confirmation",
                "exit": END
            }
        )
        self.workflow.add_conditional_edges("wait_for_confirmation", self.check_plan_confirmed,
            {
                "confirmed": "dispatch_to_agent",
                "needs_confirmation": "create_plan",  # Loop back to adjust plan
                "exit": END
            }
        )
        
        # Execution flow
        self.workflow.add_edge("dispatch_to_agent", "interact_with_agent")
        self.workflow.add_edge("interact_with_agent", "present_results")
        self.workflow.add_edge("present_results", "user_input")

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
    
    def _setup_agent_collaboration(self):
        """Setup inter-agent collaboration by passing agent references."""
        # Create agent registry for sub-agents to access each other
        agent_registry = {
            "coder": self.coder_agent,
            "file_manager": self.file_manager_agent,
            "log_analyzer": self.log_analyzer_agent,
            "general": self.general_agent,
            "orchestrator": self
        }
        
        # Pass registry to each agent
        if hasattr(self.coder_agent, 'set_agent_registry'):
            self.coder_agent.set_agent_registry(agent_registry)
        if hasattr(self.file_manager_agent, 'set_agent_registry'):
            self.file_manager_agent.set_agent_registry(agent_registry)
        if hasattr(self.log_analyzer_agent, 'set_agent_registry'):
            self.log_analyzer_agent.set_agent_registry(agent_registry)
        if hasattr(self.general_agent, 'set_agent_registry'):
            self.general_agent.set_agent_registry(agent_registry)
    
    async def ask_fresh_start(self):
        """
        Ask user if they want to start fresh with a new conversation.
        If yes, delete memory from checkpoints.db; otherwise continue previous conversation.
        """
        self.console.print(
            Panel.fit(
                Markdown("Welcome to the Orchestrator Agent!"),
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
                db_path = os.path.join(os.getcwd(), "orchestrator_checkpoints.db")
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

        print("🔄 Initializing Orchestrator Agent...")

        # Initialize sub-agents
        config_path = "config.json"  # Use same config file
        
        # Create agents (they will change to working directory during their own initialization)
        self.coder_agent = CoderAgent(config_path)
        self.file_manager_agent = Agent(config_path)
        self.log_analyzer_agent = LogAnalyzerAgent(config_path)
        self.general_agent = GeneralAssistantAgent(config_path)
        
        # Pass agent references to each sub-agent for inter-agent collaboration BEFORE initialization
        # This allows agents to have collaboration tools available during initialization
        self._setup_agent_collaboration()
        
        # Initialize the sub-agents (after setting up collaboration)
        # Each agent will change to the working directory during initialization
        await self.coder_agent.initialize()
        await self.file_manager_agent.initialize()
        await self.log_analyzer_agent.initialize()
        await self.general_agent.initialize()
        
        
        self._initialized = True

        # Compile graph
        agent_config = self.config.get("agents", {}).get("orchestrator", {})
        db_path = os.path.join(os.getcwd(), agent_config.get("checkpoint_db", "orchestrator_checkpoints.db"))
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)

        # Optional: print a greeting panel
        self.console.print(
            Panel.fit(
                Markdown("**Orchestrator Agent** - Your AI assistant coordinator"),
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
        agent_config = self.config.get("agents", {}).get("orchestrator", {})
        config = {
            "configurable": {"thread_id": agent_config.get("thread_id", "orchestrator-1")},
            "recursion_limit": self.config.get("ui", {}).get("recursion_limit", 100)
        }
        initial_state = {"messages": AIMessage(content="I'm your orchestrator agent. I can help you with coding tasks, file management, and log analysis. What would you like to do today?")}

        try:
            await self.agent.ainvoke(initial_state, config=config)
            self.console.print(
                Panel.fit(
                    Markdown("Session ended!"),
                    title="Bye",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            self.console.print(
                Panel.fit(
                    Markdown("Session interrupted by user!"),
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

    def route_request(self, state: OrchestratorAgentState) -> str:
        """
        Route the request based on the active agent.
        If no active agent, analyze the request to select one.
        If an agent is already active, continue interacting with it.
        """
        last_message = state.messages[-1]
        
        # Check for exit command
        if hasattr(last_message, "content") and last_message.content == "__EXIT__":
            return "exit"
            
        # Check for switch agent command
        if hasattr(last_message, "content") and last_message.content.lower() in ["switch", "change agent", "new task"]:
            return "analyze"
        
        # If no active agent, analyze the request
        if state.active_agent == "orchestrator":
            return "analyze"
        
        # Otherwise, continue with the active agent
        return "interact"

    def check_need_plan(self, state: OrchestratorAgentState) -> str:
        try:
            if state.selected_agent == "general":
                return "no_plan"
            return "needs_plan"
        except Exception:
            return "needs_plan"

    # Node: user_input
    def user_input(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Ask user for input and append HumanMessage to state.
        Handles empty input by asking again.
        Supports exit commands to quit the conversation.
        Supports switch commands to change agents.
        """
        while True:
            # Show current active agent if not the orchestrator
            if state.active_agent != "orchestrator":
                self.console.print(f"[bold yellow]Currently interacting with {state.active_agent} agent[/bold yellow]")
                self.console.print("[dim]Type 'switch' to change agents or 'exit' to quit[/dim]")
            
            self.console.print("[bold cyan]User Input[/bold cyan]: ")
            user_input = self.console.input("> ").strip()

            # check empty input
            if not user_input:
                self.console.print("Empty input is not allowed. Please try again.")
                self.console.print("Tip: Type 'exit', 'bye' or 'quit' to end the conversation.\n")
                continue
                
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                usage = tracker.summary()
                usage_md = f"Total tokens: {usage.get('total', 0)}\nPrompt: {usage.get('total_prompt', 0)}\nCompletion: {usage.get('total_completion', 0)}"
                per_model = usage.get("per_model", {})
                if per_model:
                    lines = ["Per model:"]
                    for k, v in per_model.items():
                        lines.append(f"- {k}: total={v.get('total', 0)}, prompt={v.get('prompt', 0)}, completion={v.get('completion', 0)}")
                    usage_md = usage_md + "\n" + "\n".join(lines)
                self.console.print(
                    Panel.fit(
                        Markdown(f"**Session Ended**\n\n{usage_md}"),
                        title="[bold green]Token Usage[/bold green]",
                        border_style="green",
                    )
                )
                self.console.print(
                    Panel.fit(
                        Markdown("Thank you for using Orchestrator Agent, good bye!"),
                        title="Conversation ended",
                        border_style="cyan",
                    )
                )
                return {"messages": [HumanMessage(content="__EXIT__")]}
                
            if user_input.strip().lower() in ["switch", "change agent", "new task"]:
                self.console.print("[bold yellow]Switching agents...[/bold yellow]")
                return {"messages": [HumanMessage(content=user_input)], "active_agent": "orchestrator"}
                
            return {"messages": [HumanMessage(content=user_input)]}

    # Node: analyze_request
    def analyze_request(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Analyze the user's request to determine which sub-agent to use.
        """
        system_text = """
        You are an orchestrator agent that analyzes user requests and determines which specialized agent to use.
        
        You have four specialized agents available:
        1. Coder Agent - Handles software development tasks including:
           - Writing and analyzing code (especially Python and PowerShell)
           - Debugging and testing
           - Code optimization and refactoring
           - Running code execution
           - Software architecture questions
           - Creating scripts for Windows OS automation
        
        2. File Manager Agent - Handles file and document management tasks including:
           - Searching for files and directories
           - Reading file contents
           - Writing and creating files
           - Managing file systems
           - Document analysis
        
        3. Log Analyzer Agent - Handles log analysis and troubleshooting tasks including:
            - Searching log files for patterns and errors
            - Analyzing log files for root causes
            - Windows Event Log analysis
            - Troubleshooting Windows OS issues
            - Summarizing log findings
        
        4. General Assistant - Handles general-purpose requests when none of the specialized agents clearly apply; can consult other agents when needed
        
        Analyze the user's request and determine which agent would be most appropriate.
        Respond with ONLY one of the following:
        - "coder" if the request is primarily about coding, programming, or software development
        - "file_manager" if the request is primarily about file management, document handling, or file system operations
        - "log_analyzer" if the request is about log analysis, troubleshooting, or identifying issues from logs
        - "general" if the request is general-purpose or unclear which specialized agent should handle it
        
        Consider the primary intent of the request, not just keywords.
        """
        
        # Get the last user message
        last_message = state.messages[-1]
        
        # Compose messages for analysis
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Analyze this request: {last_message.content}")
        ]

        # Invoke model with thinking indicator
        from rich.spinner import Spinner
        from rich.live import Live
        with Live(Spinner("point", "[bold magenta]Analyzing request...[/bold magenta]"), refresh_per_second=10) as live:
            response = self.model.invoke(messages)
            live.stop()
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
            tracker.record("orchestrator", tracker.estimate_tokens(prompt_text), tracker.estimate_tokens(completion_text))
        except Exception:
            pass
        
        # Extract the agent selection from the response
        selected_agent = response.content.strip().lower()
        if selected_agent not in ["coder", "file_manager", "log_analyzer", "general"]:
            selected_agent = "general"
            self.console.print(f"[yellow]Request unclear, defaulting to general assistant[/yellow]")
        else:
            self.console.print(f"[green]Selected {selected_agent} agent for this request[/green]")
        
        return {"selected_agent": selected_agent}

    # Node: create_plan
    def create_plan(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Create an execution plan based on the analyzed request.
        """
        selected_agent = state.selected_agent
        last_message = state.messages[-1]
        
        # Include plan context if modifying
        plan_context = ""
        if state.plan:
            plan_context = f"\n\nPrevious plan (if modifying):\n{state.plan}"
        
        system_text = f"""
        You are an orchestrator agent creating an execution plan for a user request.
        
        The user request is: {last_message.content if hasattr(last_message, 'content') else str(last_message)}
        
        You have selected the {selected_agent} agent to handle this task.
        {plan_context}
        
        Create a detailed execution plan that includes:
        1. **Objective**: Clear statement of what needs to be accomplished
        2. **Steps**: Step-by-step breakdown of how to complete the task
        3. **Agents Involved**: Which agent(s) will be used (may need multiple agents for collaboration)
        4. **Expected Outcomes**: What results are expected
        5. **Potential Issues**: Any potential problems or considerations
        
        Format the plan clearly with sections and bullet points.
        If the task requires multiple agents, specify how they will collaborate.
        Be specific about which agent does what and when collaboration is needed.
        """
        
        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=f"Create an execution plan for: {last_message.content if hasattr(last_message, 'content') else str(last_message)}")
        ]
        
        from rich.spinner import Spinner
        from rich.live import Live
        with Live(Spinner("point", "[bold magenta]Creating execution plan...[/bold magenta]"), refresh_per_second=10) as live:
            response = self.model.invoke(messages)
            live.stop()
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
            tracker.record("orchestrator", tracker.estimate_tokens(prompt_text), tracker.estimate_tokens(completion_text))
        except Exception:
            pass
        
        plan = response.content.strip()
        
        # Display the plan to user
        self.console.print(
            Panel.fit(
                Markdown(plan),
                title="[bold cyan]Execution Plan[/bold cyan]",
                border_style="cyan",
            )
        )
        
        return {"plan": plan, "plan_confirmed": False}

    # Node: wait_for_confirmation
    def wait_for_confirmation(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Wait for user confirmation or modification of the plan.
        """
        self.console.print("\n[bold yellow]Please review the plan above.[/bold yellow]")
        self.console.print("[dim]Options:[/dim]")
        self.console.print("[dim]- Type 'yes', 'y', 'confirm', or 'ok' to proceed with the plan[/dim]")
        self.console.print("[dim]- Type 'modify' or 'change' followed by your modifications[/dim]")
        self.console.print("[dim]- Type 'cancel' to abort[/dim]\n")
        
        while True:
            user_input = self.console.input("[bold cyan]Your response:[/bold cyan] ").strip()
            
            if not user_input:
                continue
            
            user_lower = user_input.lower()
            
            # Check for confirmation
            if user_lower in ["yes", "y", "confirm", "ok", "proceed", "execute"]:
                self.console.print("[green]✓ Plan confirmed. Proceeding with execution...[/green]\n")
                return {"plan_confirmed": True, "messages": [HumanMessage(content="__PLAN_CONFIRMED__")]}
            
            # Check for cancellation
            elif user_lower in ["cancel", "abort", "stop"]:
                self.console.print("[yellow]Plan cancelled by user.[/yellow]\n")
                return {"plan_confirmed": False, "messages": [HumanMessage(content="__PLAN_CANCELLED__")]}
            
            # Check for modification
            elif user_lower.startswith("modify") or user_lower.startswith("change") or user_lower.startswith("update"):
                modification = user_input[len("modify"):].strip() or user_input[len("change"):].strip() or user_input[len("update"):].strip()
                if not modification:
                    modification = self.console.input("[bold cyan]Please describe your modifications:[/bold cyan] ").strip()
                
                # Update plan based on modification
                system_text = f"""
                The user wants to modify the execution plan. 
                
                Current plan:
                {state.plan}
                
                User's modification request:
                {modification}
                
                Update the plan according to the user's request. Return the complete updated plan.
                """
                
                messages = [
                    SystemMessage(content=system_text),
                    HumanMessage(content=f"Update the plan: {modification}")
                ]
                
                from rich.spinner import Spinner
                from rich.live import Live
                with Live(Spinner("point", "[bold magenta]Updating plan...[/bold magenta]"), refresh_per_second=10) as live:
                    response = self.model.invoke(messages)
                    live.stop()
                
                updated_plan = response.content.strip()
                
                self.console.print(
                    Panel.fit(
                        Markdown(updated_plan),
                        title="[bold cyan]Updated Execution Plan[/bold cyan]",
                        border_style="cyan",
                    )
                )
                
                return {"plan": updated_plan, "plan_confirmed": False}
            
            else:
                self.console.print("[yellow]Please respond with 'yes' to confirm, 'modify' to change, or 'cancel' to abort.[/yellow]\n")

    def check_plan_confirmed(self, state: OrchestratorAgentState) -> str:
        """
        Check if plan is confirmed or needs user confirmation.
        """
        last_message = state.messages[-1] if state.messages else None
        
        # Check if user cancelled
        if last_message and hasattr(last_message, "content") and last_message.content == "__PLAN_CANCELLED__":
            return "exit"
        
        # Check if plan is confirmed
        if state.plan_confirmed:
            return "confirmed"
        
        # Needs confirmation
        return "needs_confirmation"

    # Node: dispatch_to_agent
    async def dispatch_to_agent(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Initialize the selected sub-agent and set it as the active agent.
        """
        selected_agent = state.selected_agent
        
        # Find the original user request (skip confirmation messages)
        original_request = None
        for msg in reversed(state.messages):
            if (hasattr(msg, "content") and 
                isinstance(msg, HumanMessage) and 
                msg.content not in ["__PLAN_CONFIRMED__", "__PLAN_CANCELLED__"]):
                original_request = msg.content
                break
        
        # If no original request found, use the last non-confirmation message
        if not original_request:
            for msg in reversed(state.messages):
                if hasattr(msg, "content") and msg.content not in ["__PLAN_CONFIRMED__", "__PLAN_CANCELLED__"]:
                    original_request = str(msg.content)
                    break
        
        if selected_agent == "general":
            context_message = f"""Original Request: {original_request}

Please assist the user directly without creating or following a plan."""
        else:
            context_message = f"""Original Request: {original_request}

Execution Plan:
{state.plan}

Please proceed with executing this plan. The plan has been confirmed by the user."""
        
        self.console.print(f"[bold cyan]Activating {selected_agent} agent...[/bold cyan]")
        
        # Initialize agent context with full context
        agent_context = {
            "messages": [HumanMessage(content=context_message)],
            "initialized": True,
            "original_request": original_request,
            "plan": state.plan
        }
        
        return {
            "active_agent": selected_agent,
            "agent_context": agent_context
        }

    # Node: interact_with_agent
    async def interact_with_agent(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Interact with the currently active agent.
        """
        active_agent = state.active_agent
        last_message = state.messages[-1]
        
        # Get or initialize agent context
        agent_context = state.agent_context.copy() if state.agent_context else {"messages": []}
        
        # If this is the first interaction after dispatch, use the context message
        # Otherwise, add the new message to the agent's context
        if last_message.content in ["__PLAN_CONFIRMED__"]:
            # Use the context from agent_context (which has the full context with plan)
            if agent_context.get("messages"):
                current_message = agent_context["messages"][0]
            else:
                # Fallback: build context message
                original_request = agent_context.get("original_request", "Unknown request")
                plan = agent_context.get("plan", state.plan)
                context_message = f"""Original Request: {original_request}

Execution Plan:
{plan}

Please proceed with executing this plan. The plan has been confirmed by the user."""
                current_message = HumanMessage(content=context_message)
        else:
            # Regular message, add it to context
            current_message = HumanMessage(content=last_message.content)
            agent_context["messages"].append(current_message)
        
        self.console.print(f"[bold cyan]Processing with {active_agent} agent...[/bold cyan]")
        
        try:
            if active_agent == "coder":
                # Use the coder agent with its memory
                # Get the coder's memory from state or initialize if empty
                coder_memory = list(state.coder_memory) if state.coder_memory else []
                
                # Add the current message to the coder's memory
                coder_memory.append(current_message)
                
                # Create a temporary state for the coder agent with its memory
                coder_state = type('State', (), {'messages': coder_memory})()
                
                # Invoke the coder agent's model_response method
                coder_response = self.coder_agent.model_response(coder_state)
                
                # Extract the response and check for tool calls
                if "messages" in coder_response and len(coder_response["messages"]) > 0:
                    response_message = coder_response["messages"][0]
                    
                    # Update the coder_state with the model's response
                    coder_state.messages = coder_memory + [response_message]
                    
                    # Debug information
                    self.console.print(f"[dim]Response message type: {type(response_message)}[/dim]")
                    
                    # Check if the response is from the model and contains tool calls
                    if (isinstance(response_message, AIMessage) and 
                        hasattr(response_message, 'tool_calls') and 
                        response_message.tool_calls):
                        self.console.print(f"[yellow]Agent wants to use tools, executing...[/yellow]")
                        
                        # Execute the tool calls
                        tool_response = await self.coder_agent.tool_use(coder_state)
                        
                        # Add tool response to the coder's memory
                        if "messages" in tool_response:
                            # Update the coder_state with the tool responses
                            coder_state.messages = coder_memory + [response_message] + tool_response["messages"]
                            # Update the coder's memory with all messages
                            coder_memory = coder_state.messages
                        
                        # Get the final response after tool execution
                        final_response = self.coder_agent.model_response(coder_state)
                        final_message = final_response["messages"][0]
                        
                        # Extract the response content
                        if isinstance(final_message.content, list):
                            response_text = ""
                            for item in final_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = final_message.content
                        
                        # Add the final response to the coder's memory
                        final_ai_message = AIMessage(content=response_text)
                        coder_memory.append(final_ai_message)
                        
                        # Update agent context with the final response
                        agent_context["messages"].append(final_ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "coder_memory": coder_memory
                        }
                        
                        # Update the state with the coder's memory
                        state.coder_memory = coder_memory
                        
                        return result
                    else:
                        # No tool calls, just return the response
                        if isinstance(response_message.content, list):
                            response_text = ""
                            for item in response_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = response_message.content
                        
                        # Add the response to the coder's memory
                        ai_message = AIMessage(content=response_text)
                        coder_memory.append(ai_message)
                        
                        # Update agent context with the response
                        agent_context["messages"].append(ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "coder_memory": coder_memory
                        }
                        
                        # Update the state with the coder's memory
                        state.coder_memory = coder_memory
                        
                        return result
                else:
                    error_msg = "No valid response from coder agent"
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                    return {"agent_response": error_msg, "agent_context": agent_context}
                
            elif active_agent == "file_manager":
                # Use the file manager agent with its memory
                # Get the file_manager's memory from state or initialize if empty
                file_manager_memory = list(state.file_manager_memory) if state.file_manager_memory else []
                
                # Use context message if available, otherwise use last message
                if last_message.content in ["__PLAN_CONFIRMED__"]:
                    if agent_context.get("messages"):
                        current_message = agent_context["messages"][0]
                    else:
                        original_request = agent_context.get("original_request", "Unknown request")
                        plan = agent_context.get("plan", state.plan)
                        context_message = f"""Original Request: {original_request}

Execution Plan:
{plan}

Please proceed with executing this plan. The plan has been confirmed by the user."""
                        current_message = HumanMessage(content=context_message)
                else:
                    current_message = HumanMessage(content=last_message.content)
                
                file_manager_memory.append(current_message)
                
                # Create a temporary state for the file manager agent with its memory
                file_manager_state = type('State', (), {'messages': file_manager_memory})()
                
                # Invoke the file_manager agent's model_response method
                file_manager_response = self.file_manager_agent.model_response(file_manager_state)
                
                # Extract the response and check for tool calls
                if "messages" in file_manager_response and len(file_manager_response["messages"]) > 0:
                    response_message = file_manager_response["messages"][0]
                    
                    # Update the file_manager_state with the model's response
                    file_manager_state.messages = file_manager_memory + [response_message]
                    
                    # Debug information
                    self.console.print(f"[dim]Response message type: {type(response_message)}[/dim]")
                    
                    # Check if the response is from the model and contains tool calls
                    if (isinstance(response_message, AIMessage) and 
                        hasattr(response_message, 'tool_calls') and 
                        response_message.tool_calls):
                        self.console.print(f"[yellow]Agent wants to use tools, executing...[/yellow]")
                        
                        # Execute the tool calls
                        tool_response = await self.file_manager_agent.tool_use(file_manager_state)
                        
                        # Add tool response to the file_manager's memory
                        if "messages" in tool_response:
                            # Update the file_manager_state with the tool responses
                            file_manager_state.messages = file_manager_memory + [response_message] + tool_response["messages"]
                            # Update the file_manager's memory with all messages
                            file_manager_memory = file_manager_state.messages
                        
                        # Get the final response after tool execution
                        final_response = self.file_manager_agent.model_response(file_manager_state)
                        final_message = final_response["messages"][0]
                        
                        # Extract the response content
                        if isinstance(final_message.content, list):
                            response_text = ""
                            for item in final_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = final_message.content
                        
                        # Add the final response to the file_manager's memory
                        final_ai_message = AIMessage(content=response_text)
                        file_manager_memory.append(final_ai_message)
                        
                        # Update agent context with the final response
                        agent_context["messages"].append(final_ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "file_manager_memory": file_manager_memory
                        }
                        
                        # Update the state with the file_manager's memory
                        state.file_manager_memory = file_manager_memory
                        
                        return result
                    else:
                        # No tool calls, just return the response
                        if isinstance(response_message.content, list):
                            response_text = ""
                            for item in response_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = response_message.content
                        
                        # Add the response to the file_manager's memory
                        ai_message = AIMessage(content=response_text)
                        file_manager_memory.append(ai_message)
                        
                        # Update agent context with the response
                        agent_context["messages"].append(ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "file_manager_memory": file_manager_memory
                        }
                        
                        # Update the state with the file_manager's memory
                        state.file_manager_memory = file_manager_memory
                        
                        return result
                else:
                    error_msg = "No valid response from file_manager agent"
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                    return {"agent_response": error_msg, "agent_context": agent_context}
                
            elif active_agent == "log_analyzer":
                # Use the log analyzer agent with its memory
                log_analyzer_memory = list(state.log_analyzer_memory) if state.log_analyzer_memory else []
                
                # Use context message if available, otherwise use last message
                if last_message.content in ["__PLAN_CONFIRMED__"]:
                    if agent_context.get("messages"):
                        current_message = agent_context["messages"][0]
                    else:
                        original_request = agent_context.get("original_request", "Unknown request")
                        plan = agent_context.get("plan", state.plan)
                        context_message = f"""Original Request: {original_request}

Execution Plan:
{plan}

Please proceed with executing this plan. The plan has been confirmed by the user."""
                        current_message = HumanMessage(content=context_message)
                else:
                    current_message = HumanMessage(content=last_message.content)
                
                log_analyzer_memory.append(current_message)
                
                # Create a temporary state for the log analyzer agent with its memory
                log_analyzer_state = type('State', (), {'messages': log_analyzer_memory})()
                
                # Invoke the log_analyzer agent's model_response method
                log_analyzer_response = self.log_analyzer_agent.model_response(log_analyzer_state)
                
                # Extract the response and check for tool calls
                if "messages" in log_analyzer_response and len(log_analyzer_response["messages"]) > 0:
                    response_message = log_analyzer_response["messages"][0]
                    
                    # Update the log_analyzer_state with the model's response
                    log_analyzer_state.messages = log_analyzer_memory + [response_message]
                    
                    # Check if the response is from the model and contains tool calls
                    if (isinstance(response_message, AIMessage) and 
                        hasattr(response_message, 'tool_calls') and 
                        response_message.tool_calls):
                        self.console.print(f"[yellow]Agent wants to use tools, executing...[/yellow]")
                        
                        # Execute the tool calls
                        tool_response = await self.log_analyzer_agent.tool_use(log_analyzer_state)
                        
                        # Add tool response to the log_analyzer's memory
                        if "messages" in tool_response:
                            log_analyzer_state.messages = log_analyzer_memory + [response_message] + tool_response["messages"]
                            log_analyzer_memory = log_analyzer_state.messages
                        
                        # Get the final response after tool execution
                        final_response = self.log_analyzer_agent.model_response(log_analyzer_state)
                        final_message = final_response["messages"][0]
                        
                        # Extract the response content
                        if isinstance(final_message.content, list):
                            response_text = ""
                            for item in final_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = final_message.content
                        
                        # Add the final response to the log_analyzer's memory
                        final_ai_message = AIMessage(content=response_text)
                        log_analyzer_memory.append(final_ai_message)
                        
                        # Update agent context with the final response
                        agent_context["messages"].append(final_ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "log_analyzer_memory": log_analyzer_memory
                        }
                        
                        # Update the state with the log_analyzer's memory
                        state.log_analyzer_memory = log_analyzer_memory
                        
                        return result
                    else:
                        # No tool calls, just return the response
                        if isinstance(response_message.content, list):
                            response_text = ""
                            for item in response_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = response_message.content
                        
                        # Add the response to the log_analyzer's memory
                        ai_message = AIMessage(content=response_text)
                        log_analyzer_memory.append(ai_message)
                        
                        # Update agent context with the response
                        agent_context["messages"].append(ai_message)
                        
                        # Create result with agent response, context and memory
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "log_analyzer_memory": log_analyzer_memory
                        }
                        
                        # Update the state with the log_analyzer's memory
                        state.log_analyzer_memory = log_analyzer_memory
                        
                        return result
                else:
                    error_msg = "No valid response from log_analyzer agent"
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                    return {"agent_response": error_msg, "agent_context": agent_context}
            elif active_agent == "general":
                general_memory = list(state.general_memory) if state.general_memory else []
                current_message = HumanMessage(content=last_message.content)
                general_memory.append(current_message)
                general_state = type('State', (), {'messages': general_memory})()
                general_response = self.general_agent.model_response(general_state)
                if "messages" in general_response and len(general_response["messages"]) > 0:
                    response_message = general_response["messages"][0]
                    general_state.messages = general_memory + [response_message]
                    if (isinstance(response_message, AIMessage) and hasattr(response_message, 'tool_calls') and response_message.tool_calls):
                        tool_response = await self.general_agent.tool_use(general_state)
                        if "messages" in tool_response:
                            general_state.messages = general_memory + [response_message] + tool_response["messages"]
                            general_memory = general_state.messages
                        final_response = self.general_agent.model_response(general_state)
                        final_message = final_response["messages"][0]
                        if isinstance(final_message.content, list):
                            response_text = ""
                            for item in final_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = final_message.content
                        final_ai_message = AIMessage(content=response_text)
                        general_memory.append(final_ai_message)
                        agent_context["messages"].append(final_ai_message)
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "general_memory": general_memory
                        }
                        state.general_memory = general_memory
                        return result
                    else:
                        if isinstance(response_message.content, list):
                            response_text = ""
                            for item in response_message.content:
                                if item["type"] == "text":
                                    response_text += item.get("text", "")
                        else:
                            response_text = response_message.content
                        ai_message = AIMessage(content=response_text)
                        general_memory.append(ai_message)
                        agent_context["messages"].append(ai_message)
                        result = {
                            "agent_response": response_text,
                            "agent_context": agent_context,
                            "general_memory": general_memory
                        }
                        state.general_memory = general_memory
                        return result
                else:
                    error_msg = "No valid response from general agent"
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                    return {"agent_response": error_msg, "agent_context": agent_context}
                
        except Exception as e:
            error_msg = f"Error occurred while processing with {active_agent} agent: {str(e)}"
            self.console.print(f"[bold red]{error_msg}[/bold red]")
            return {"agent_response": error_msg}

    # Node: present_results
    def present_results(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Present the results from the sub-agent to the user.
        """
        agent_response = state.agent_response
        active_agent = state.active_agent
        
        self.console.print(
            Panel.fit(
                Markdown(agent_response),
                title=f"[bold cyan]Response from {active_agent} agent[/bold cyan]",
                border_style="cyan",
            )
        )

        return {"messages": [AIMessage(content=agent_response)]}

    def print_mermaid_workflow(self):
        """
        Utility: print Mermaid diagram to visualize the graph edges.
        """
        try:
            mermaid = self.agent.get_graph().draw_mermaid_png(
                output_file_path="orchestrator_workflow.png",
                max_retries=5,
                retry_delay=2,
            )
        except Exception as e:
            print(f"Error generating mermaid PNG: {e}")
            mermaid = self.agent.get_graph().draw_mermaid()
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Orchestrator Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            print(self.agent.get_graph().draw_ascii())