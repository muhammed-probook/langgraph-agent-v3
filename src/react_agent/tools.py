
"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from pydantic import BaseModel

from react_agent.configuration import Configuration

class Appointment(BaseModel):
    """Model representing an appointment."""
    id: int
    time: str
    description: str
    
def cancel_appointment(
    appointment_id: int,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Cancel an appointment."""
    return f"Appointment {appointment_id} cancelled."


def get_appointments(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> List[Appointment]:
    """Retrieve a list of appointments for the current user.
    
    This function fetches all scheduled appointments for the user. In a real system,
    this would query a database or external service based on user authentication.
    
    Args:
        config: Configuration for the tool, automatically injected.
        
    Returns:
        List[Appointment]: A list of appointment objects containing id, time, and description.
    """

    # In a real system, this would query a database
    appointments = [
        Appointment(
            id=1,
            time="2025-01-01 10:00:00",
            description="Home cleaning service"
        ),
        Appointment(
            id=2,
            time="2025-01-05 14:00:00",
            description="Plumbing repair"
        )
    ]
    return appointments


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search, get_appointments]



