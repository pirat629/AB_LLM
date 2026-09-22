from pydantic import BaseModel

TOOLS = {}

def register(fn, args_model: type[BaseModel], description: str):
    TOOLS[fn.__name__] = {
        "fn": fn,
        "args": args_model,
        "schema": {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": args_model.model_json_schema()
            }
        }
    }