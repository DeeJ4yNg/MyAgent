from typing import Annotated, Sequence, Dict, Any, List
from dotenv import load_dotenv
import os
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

# Import sub-agents
from coder_agent import CoderAgent
from agent import Agent


class OrchestratorAgentState(BaseModel):
    """
    Persistent agent state tracked across the graph.
    - messages: complete chat history (system + user + assistant + tool messages)
    - selected_agent: which sub-agent to use (coder or file_manager)
    - agent_response: response from the sub-agent
    - active_agent: which agent is currently active (orchestrator, coder, or file_manager)
    - agent_context: context for the currently active agent
    - coder_memory: conversation history with the coder agent
    - file_manager_memory: conversation history with the file_manager agent
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    selected_agent: str = ""
    agent_response: str = ""
    active_agent: str = "orchestrator"  # Default to orchestrator
    agent_context: Dict[str, Any] = {}  # Store context for active agent
    coder_memory: List[BaseMessage] = []  # Store conversation history with coder agent
    file_manager_memory: List[BaseMessage] = []  # Store conversation history with file_manager agent


class OrchestratorAgent:
    def __init__(self):
        self._initialized = False
        # Load environment
        load_dotenv()
        
        # Rich console for UI
        self.console = Console()
        
        # Let user choose model type
        self.model_type = self._choose_model_type()
        
        # Initialize model based on user choice
        if self.model_type == "ollama":
            # Get Ollama configuration from environment
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            
            # Model instantiation (Ollama)
            self.model = ChatOllama(
                base_url=ollama_base_url,
                model=ollama_model,
                temperature=0.2,  # Lower temperature for more consistent planning
                max_tokens=4096,
            )
            self.console.print(f"[green]Using Ollama model: {ollama_model} at {ollama_base_url}[/green]")
        else:  # cloud API
            # Get OpenAI configuration from environment
            api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("OPENAI_MODEL_ORC", "gpt-4")
            
            if not api_key:
                self.console.print("[red]Error: OPENAI_API_KEY environment variable is not set![/red]")
                self.console.print("[yellow]Please set the OPENAI_API_KEY in your .env file.[/yellow]")
                exit(1)
            
            # Model instantiation (OpenAI API)
            self.model = ChatOpenAI(
                base_url=api_base,
                api_key=api_key,
                model=model_name,
                temperature=0.2,  # Lower temperature for more consistent planning
                max_tokens=4096,
            )
            self.console.print(f"[green]Using Cloud API model: {model_name} at {api_base}[/green]")

        # Build workflow graph
        self.workflow = StateGraph(OrchestratorAgentState)

        # Register nodes
        self.workflow.add_node("user_input", self.user_input)
        self.workflow.add_node("analyze_request", self.analyze_request)
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
        
        self.workflow.add_edge("analyze_request", "dispatch_to_agent")
        self.workflow.add_edge("dispatch_to_agent", "interact_with_agent")
        self.workflow.add_edge("interact_with_agent", "present_results")
        self.workflow.add_edge("present_results", "user_input")

    def _choose_model_type(self) -> str:
        """
        Let user choose between Ollama and Cloud API models
        
        Returns:
            str: 'ollama' or 'cloud'
        """
        self.console.print(
            Panel.fit(
                Markdown("# Model Selection\n\nPlease choose which model to use:"),
                title="[bold blue]Model Selection[/bold blue]",
                border_style="blue",
            )
        )
        
        while True:
            self.console.print("[bold cyan]1.[/bold cyan] [green]Ollama[/green] - Local model deployment")
            self.console.print("[bold cyan]2.[/bold cyan] [green]Cloud API[/green] - Remote model via OpenAI API format")
            
            choice = self.console.input("\n[bold cyan]Enter your choice (1 or 2):[/bold cyan] ").strip()
            
            if choice == "1":
                return "ollama"
            elif choice == "2":
                return "cloud"
            else:
                self.console.print("[bold red]Invalid choice. Please enter 1 or 2.[/bold red]")
    
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
        self.coder_agent = CoderAgent()
        self.file_manager_agent = Agent()
        
        # Initialize the sub-agents
        await self.coder_agent.initialize()
        await self.file_manager_agent.initialize()
        
        self._initialized = True

        # Compile graph
        db_path = os.path.join(os.getcwd(), "orchestrator_checkpoints.db")
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
        config = {"configurable": {"thread_id": "orchestrator-1"}, "recursion_limit": 100}
        initial_state = {"messages": AIMessage(content="I'm your orchestrator agent. I can help you with coding tasks and file management. What would you like to do today?")}

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
        
        You have two specialized agents available:
        1. Coder Agent - Handles software development tasks including:
           - Writing and analyzing code
           - Debugging and testing
           - Code optimization and refactoring
           - Running code execution
           - Software architecture questions
        
        2. File Manager Agent - Handles file and document management tasks including:
           - Searching for files and directories
           - Reading file contents
           - Writing and creating files
           - Managing file systems
           - Document analysis
        
        Analyze the user's request and determine which agent would be most appropriate.
        Respond with ONLY one of the following:
        - "coder" if the request is primarily about coding, programming, or software development
        - "file_manager" if the request is primarily about file management, document handling, or file system operations
        
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
        
        # Extract the agent selection from the response
        selected_agent = response.content.strip().lower()
        if selected_agent not in ["coder", "file_manager"]:
            # Default to file_manager if unclear
            selected_agent = "file_manager"
            self.console.print(f"[yellow]Request unclear, defaulting to file manager agent[/yellow]")
        else:
            self.console.print(f"[green]Selected {selected_agent} agent for this request[/green]")
        
        return {"selected_agent": selected_agent}

    # Node: dispatch_to_agent
    async def dispatch_to_agent(self, state: OrchestratorAgentState) -> OrchestratorAgentState:
        """
        Initialize the selected sub-agent and set it as the active agent.
        """
        selected_agent = state.selected_agent
        last_message = state.messages[-1]
        
        self.console.print(f"[bold cyan]Activating {selected_agent} agent...[/bold cyan]")
        
        # Initialize agent context
        agent_context = {
            "messages": [HumanMessage(content=last_message.content)],
            "initialized": True
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
        
        # Add the new message to the agent's context
        agent_context["messages"].append(HumanMessage(content=last_message.content))
        
        self.console.print(f"[bold cyan]Processing with {active_agent} agent...[/bold cyan]")
        
        try:
            if active_agent == "coder":
                # Use the coder agent with its memory
                # Get the coder's memory from state or initialize if empty
                coder_memory = list(state.coder_memory) if state.coder_memory else []
                
                # Add the current message to the coder's memory
                current_message = HumanMessage(content=last_message.content)
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
                
                # Add the current message to the file_manager's memory
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