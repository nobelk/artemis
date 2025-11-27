from qwen_agent.tools.base import BaseTool, register_tool
import json

@register_tool('calculator')
class Calculator(BaseTool):
    description = 'Perform basic arithmetic operations'
    parameters = [
        {'name': 'operation', 'type': 'string', 'required': True},
        {'name': 'a', 'type': 'number', 'required': True},
        {'name': 'b', 'type': 'number', 'required': True}
    ]

    def call(self, params: str, **kwargs) -> str:
        data = json.loads(params)
        operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else 'Error: Division by zero'
        }
        result = operations.get(data['operation'], lambda a, b: 'Invalid operation')(
            data['a'], data['b']
        )
        return json.dumps({'result': result})