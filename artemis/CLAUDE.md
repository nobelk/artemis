# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Artemis is an **Agent Reputation & Trust Environment for Macro-scale Interaction Simulation**. It's a Python-based simulation framework for AI agent interactions built on the Qwen Agent framework.

## Development Commands

### Environment Setup
- Python version: 3.13.4+
- Package manager: `uv` (recommended) or `pip`
- Install dependencies: `uv sync` or `pip install -e .`

### Running the Application
- Main application: `python main.py`
  - Interactive chat interface with an AI assistant
  - Type 'exit' or 'quit' to terminate
- Test initialization: `python test_run.py`

### Testing
- Run all tests: `pytest`
- Run tests with verbose output: `pytest -v`
- Run specific test file: `pytest tests/test_filename.py`

### Code Formatting
- Format code: `black .`
- Sort imports: `isort .`
- Format and sort together: `isort . && black .`

## Architecture

### Core Components

**Agent System** (`src/agent/`)
- `assistant.py`: Creates and configures AI assistants using Qwen Agent
- Default model: `qwen2.5:1.5b` running via local Ollama server at `http://localhost:11434/v1`
- Agents are created with a function list and system message configuration

**Tools/Functions** (`src/functions/`)
- Tools extend `BaseTool` from `qwen_agent.tools.base`
- Register tools with `@register_tool('tool_name')` decorator
- Tools must implement:
  - `description`: String describing the tool
  - `parameters`: List of dicts defining name, type, and required status
  - `call(params: str, **kwargs) -> str`: Main execution method that receives JSON string params

**Main Application** (`main.py`)
- Interactive message loop that maintains conversation history
- Streams responses from the agent using `bot.run(messages=messages)`
- Messages follow the standard role/content format: `{'role': 'user', 'content': query}`

### Key Dependencies

- **qwen-agent**: Core agent framework providing Assistant and tool infrastructure
- **python-dateutil**: Date/time utilities
- **pytest**: Testing framework
- **black**: Code formatter
- **isort**: Import sorter

### Project Structure

```
artemis/
├── src/
│   ├── agent/          # Agent configuration and creation
│   └── functions/      # Tool implementations (calculator, etc.)
├── tests/              # Test suite
├── main.py             # Interactive chat application
└── test_run.py         # Initialization verification script
```

## Adding New Tools

1. Create a new file in `src/functions/`
2. Import `BaseTool` and `register_tool` from `qwen_agent.tools.base`
3. Define a class decorated with `@register_tool('tool_name')`
4. Implement `description`, `parameters`, and `call()` method
5. Import the tool class in `src/agent/assistant.py`
6. Add the tool name to the `function_list` in `create_assistant()`

Example pattern (see `src/functions/calculator.py:4-24`):
```python
@register_tool('tool_name')
class ToolName(BaseTool):
    description = 'Tool description'
    parameters = [{'name': 'param', 'type': 'string', 'required': True}]

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params)
        # Implementation
        return json.dumps({'result': result})
```

## Local LLM Server

The project uses Ollama for local LLM inference. Ensure Ollama is running before starting the application:
- Server endpoint: `http://localhost:11434/v1`
- Default model: `qwen2.5:1.5b`
- API key configured as 'EMPTY' for local server

To change the model or server, modify the `llm_cfg` in `src/agent/assistant.py:8-12`.
