import sys
from a3.agent import BaseAgent
from agents.a1agent.a1agent import A1Agent

def main() -> None:
    if len(sys.argv) == 1:
        BaseAgent.get_agent().start_test(A1Agent())
    elif len(sys.argv) == 2:
        BaseAgent.get_agent().start_with_group_name(sys.argv[1], A1Agent())
    else:
        print("Agent: Usage: python3 agents/a1agent/__main__.py <groupname>")

if __name__ == "__main__":
    main()