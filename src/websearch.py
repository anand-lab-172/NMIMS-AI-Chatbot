from duckduckgo_search import DDGS

def web_search(query, max_results=3):

    results = []

    with DDGS() as ddgs:

        search_results = ddgs.text(query, max_results=max_results)

        for r in search_results:

            results.append(
                f"Title: {r['title']}\n"
                f"Body: {r['body']}\n"
            )

    return "\n\n".join(results)