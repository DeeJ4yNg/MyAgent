from file_management_agent import Agent
import asyncio
import sys


async def async_main():
    agent = Agent()
    try:
        await agent.ask_fresh_start()
        await agent.initialize()
        agent.print_mermaid_workflow()
        await agent.run()
    except KeyboardInterrupt:
        print("Agent interrupted by user.")
    except Exception as e:
        print(f"Agent encountered an error: {str(e)}")
        sys.exit(1)
    finally:
        await agent.close_checkpointer()


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("Main interrupted by user.")
        sys.exit(0)
