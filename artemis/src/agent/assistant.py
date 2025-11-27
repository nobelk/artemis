from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant

from src.functions.calculator import Calculator


# Create agent with multiple tools
llm_cfg = {
    'model': 'qwen2.5:1.5b',
    'model_server': 'http://localhost:11434/v1',
    'api_key': 'EMPTY',
    'generate_cfg': {
        'top_p': 0.8,              # Nucleus sampling
        'temperature': 0.7,         # Creativity control
        'max_input_tokens': 4096,   # Context window
    }
}

def create_assistant():
    multi_tool_agent = Assistant(
        llm=llm_cfg,
        function_list=['calculator'],
        system_message='You are a helpful assistant with access calculator tools.'
    )
    return multi_tool_agent

