def standard_coding_prompt(problem_description: str) -> str:
    return f"""You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).\n{problem_description}."""

def instruct_coding_prompt(problem_description: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nComplete the following Python code without any tests or explanation.\n{problem_description}\n\n### Response:"""

def detailed_coding_prompt(problem_description: str) -> str:
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{problem_description}\n\n### Requirements:\n1. Complete the code without any tests or explanation.\n2. Include all necessary data structures and imports within the response.\n3. Return executable code, wrapped in backtick."""