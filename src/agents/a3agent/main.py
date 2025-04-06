import sys
from a3.agent import BaseAgent
from a3.agent.brain import Brain
from aegis import MOVE, END_TURN, Direction, SEND_MESSAGE_RESULT, OBSERVE_RESULT, SAVE_SURV_RESULT, PREDICT_RESULT

# Create a complete Brain implementation that uses BaseAgent
class BaseAgentBrain(Brain):
    def __init__(self) -> None:
        super().__init__()
        self._agent = BaseAgent.get_agent()
    
    def handle_send_message_result(self, smr: SEND_MESSAGE_RESULT) -> None:
        self._agent.log(f"Message result: {smr}")
    
    def handle_observe_result(self, ovr: OBSERVE_RESULT) -> None:
        self._agent.log(f"Observe result: {ovr}")
    
    def handle_save_surv_result(self, ssr: SAVE_SURV_RESULT) -> None:
        self._agent.log(f"Save survivor result: {ssr}")
    
    def handle_predict_result(self, prd: PREDICT_RESULT) -> None:
        self._agent.log(f"Prediction result: {prd}")
    
    def think(self) -> None:
        self._agent.log("Simple agent thinking...")
        self._agent.send(MOVE(Direction.CENTER))
        self._agent.send(END_TURN())

def main() -> None:
    if len(sys.argv) == 1:
        BaseAgent.get_agent().start_test(BaseAgentBrain())
    elif len(sys.argv) == 2:
        BaseAgent.get_agent().start_with_group_name(sys.argv[1], BaseAgentBrain())
    else:
        print("Agent: Usage: python3 agents/a3agent/main.py <groupname>")

if __name__ == "__main__":
    main()