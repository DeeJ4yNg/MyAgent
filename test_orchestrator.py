#!/usr/bin/env python3
"""
Simple test script for the Orchestrator Agent.
This script tests the basic functionality of the orchestrator agent.
"""

import asyncio
import sys
from langchain_core.messages import HumanMessage
from orchestrator_agent import OrchestratorAgent


async def test_initialization():
    """
    Test the initialization of the orchestrator agent.
    """
    print("Testing orchestrator agent initialization...")
    
    # Create orchestrator agent instance
    orchestrator = OrchestratorAgent()
    
    # Initialize the agent
    await orchestrator.initialize()
    
    print("✅ Orchestrator agent initialized successfully!")
    
    # Test request analysis
    test_requests = [
        "Write a Python function to calculate factorial",
        "Find all .txt files in the current directory",
        "Debug this code that has a syntax error",
        "Read the contents of README.md"
    ]
    
    for request in test_requests:
        print(f"\nTesting request: '{request}'")
        
        # Create a test state
        test_state = type('State', (), {'messages': [HumanMessage(content=request)]})()
        
        # Analyze the request
        result = orchestrator.analyze_request(test_state)
        selected_agent = result["selected_agent"]
        
        print(f"Selected agent: {selected_agent}")
    
    print("\n✅ All tests completed successfully!")
    
    # Clean up
    await orchestrator.close_checkpointer()


if __name__ == "__main__":
    try:
        asyncio.run(test_initialization())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)