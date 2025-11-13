#!/usr/bin/env python3
"""
Main entry point for the Orchestrator Agent.
This script initializes and runs the orchestrator agent that coordinates
between the coder agent and file management agent.
"""

import asyncio
import sys
from orchestrator_agent import OrchestratorAgent


async def main():
    """
    Main function to initialize and run the orchestrator agent.
    """
    # Create orchestrator agent instance
    orchestrator = OrchestratorAgent()
    
    # Ask if user wants to start fresh
    await orchestrator.ask_fresh_start()
    
    # Initialize the agent
    await orchestrator.initialize()
    
    # Run the agent
    await orchestrator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)