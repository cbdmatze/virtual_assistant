import requests
import logging
from typing import Dict, Any

# Updated to ensure we use the correct variable name
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID


logger = logging.getLogger(__name__)


def test_google_api() -> bool:
    """Test the Google API setup directly"""
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q=test&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" in data:
            logger.info("Google API test successful: search results returned")
            return True
        else:
            if "error" in data:
                # Fixed: Corrected the syntax error in get method (used parentheses instead of brackets)
                logger.error(f"Google API test failed: {data['error'].get('message', 'Unknown error')}")
            else:
                logger.error(f"Google API test failed: no search results returned. Response: {data}")
            return False
    except Exception as e:
        logger.error(f"Google API test failed with error: {str(e)}")
        return False
    

def google_search(query: str) -> str:
    """
    Perform a Google search.
    
    Args:
        query: The search query
    
    Returns:
        str: The search results
    """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
    logger.info(f"Performing Google search for query: {query}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "items" in data:
            # Return a compilation of the first 5 snippets
            snippets = []
            for item in data["items"][:5]:
                # Enhanced: Include title and link for better context
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                snippets.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
                
            result = "\n\n---\n\n".join(snippets)
            logger.info(f"Google search returned {len(snippets)} results")
            return result
        
        logger.warning("Google search returned no results")
        if "error" in data:
            logger.error(f"Google search error: {data['error'].get('message', 'Unknown error')}")

        return "No search results found"
    except Exception as e:
        logger.error(f"Error performing Google search: {str(e)}")
        return f"Error performing Google search: {str(e)}"
    

def direct_google_search(query: str) -> str:
    """
    Basic function that performs a Google search and formats the results
    without requiring an LLM agent.
    """
    try:
        logger.info(f"Performing direct Google search for query: {query}")
        search_results = google_search(query)

        if not search_results or search_results == "No search results found":
            logger.warning(f"Direct Google search returned no results for query: {query}")
            return f"Sorry, I couldn't find any results for '{query}'"
        
        # Format the results for better readability
        formatted_response = f"Google Search Results for: '{query}'\n\n{search_results}"
        logger.info(f"Direct Google search successful for query: {query}")

        return formatted_response
    except Exception as e:
        logger.error(f"Error in direct Google search: {e}")
        return f"Error searching Google for '{query}': {str(e)}"


def advanced_search_with_google_api(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a more advanced Google search with structured results.
    
    Args:
        query: The search query
        num_results: Number of results to return
    
    Returns:
        Dict: Structured search results
    """
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&num={num_results}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if "items" not in data:
            logger.warning(f"Advanced Google search returned no results for: {query}")
            return {"success": False, "results": [], "error": "No results found"}
            
        processed_results = []
        for item in data["items"]:
            processed_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "displayLink": item.get("displayLink", ""),
                "pagemap": item.get("pagemap", {})
            })
           
        logger.info(f"Advanced Google search successful for: {query}")
        return {
            "success": True,
            "results": processed_results,
            "searchInformation": data.get("searchInformation", {})
        }
       
    except Exception as e:
        logger.error(f"Error in advanced Google search: {e}")
        return {"success": False, "results": [], "error": str(e)}
    