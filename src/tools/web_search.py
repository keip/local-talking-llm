from ddgs import DDGS

from . import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web using DuckDuckGo. Returns top results with titles and snippets."
    parameters = {"query": "str"}

    def execute(self, *, query: str, **_kwargs) -> str:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"- {r['title']}: {r['body']}")
            lines.append(f"  {r['href']}")
        return "\n".join(lines)
