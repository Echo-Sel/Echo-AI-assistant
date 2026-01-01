from langchain.tools import tool

@tool("calculate")
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this tool when the user asks for a calculation, e.g., "What is 25 * 48?" or "Calculate 100 / 4".
    
    Input:
    - A string containing a valid mathematical expression (e.g., "25 * 48", "100 / 4", "2 + 2").
    """
    try:
        # Evaluate the expression safely
        # Using eval with a limited scope is generally safe for simple math, 
        # but for a production system, a library like numexpr or a parser is better.
        # For this personal assistant, eval is acceptable if we trust the LLM's input.
        # We'll stick to simple eval for now as it handles standard operators well.
        allowed_names = {"abs": abs, "round": round}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed")
        
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"
