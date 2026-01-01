from langchain.tools import tool
from duckduckgo_search import DDGS

@tool("duckduckgo_search")
def duckduckgo_search_tool(query: str) -> str:
    """
    Perform a web search using DuckDuckGo and return the top results.
    Use this tool when the user asks a question that requires up-to-date information from the internet.
    
    Examples of queries:
    - "Please look up what's the weather like in Paris today?"
    - "Look up the latest tech news"
    - "yes, please search for current AI news"

    Input:
    - A natural language query string.
    """
    with DDGS() as ddgs:
        # Simplified arguments to avoid potential API issues
        results = ddgs.text(query, max_results=3)
        results_list = list(results)

    if not results_list:
        return f"Apologies, I couldn't find any results for: \"{query}\"."

    formatted_results = f"Here are the search results for: \"{query}\"\n\n"
    for i, res in enumerate(results_list, 1):
        formatted_results += (
            f"Result {i}:\n"
            f"🔹 Title: {res['title']}\n"
            f"🔗 URL: {res['href']}\n"
            f"📝 Snippet: {res['body']}\n\n"
        )
    
    return formatted_results
