import ast
import json


def extract_python_code(input_string, input_type):
    """
    """
    try:
        start_marker = "```python\n"
        end_marker = "\n```"
        start_index = input_string.find(start_marker) + len(start_marker)
        end_index = input_string.rfind(end_marker)
        parsed_string = input_string[start_index:end_index]

        parsed_obj = None
        if input_type == "list":
            parsed_obj = ast.literal_eval(parsed_string)
        elif input_type == "dict":
            parsed_obj = json.loads(parsed_string)

        return parsed_obj
            
    except Exception as e:
        raise(Exception)