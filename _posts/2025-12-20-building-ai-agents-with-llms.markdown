---
layout: post
title: "Building AI Agents with LLMs"
date: 2025-12-20 00:00:00 +0530
categories: programming
---


Artificial Intelligence has evolved beyond simple chatbots. Today, we can build AI agents — intelligent systems that can reason, plan, and take actions in the real world. These agents combine the natural language understanding of Large Language Models (LLMs) with the ability to call functions, query APIs, and interact with external systems.

In this tutorial, we’ll build a practical AI agent that monitors global disaster events using the GDACS (Global Disaster Alert and Coordination System) RSS feed. By the end, you’ll understand how to:

*   Create AI agents using LangChain
*   Extend LLMs with custom tools
*   Parse and process real-time data feeds
*   Build agents that can answer complex, data-driven questions

## What Are AI Agents?

An AI agent is an autonomous system that can:

1.  Understand natural language queries
2.  Reason about what actions to take
3.  Execute tools or functions to gather information
4.  Synthesize results into coherent responses

Unlike traditional chatbots that only respond from their training data, agents can:

*   Query live APIs
*   Search databases
*   Perform calculations
*   Interact with external services

Think of an agent as a smart assistant that can not only answer questions but also do things—like looking up real-time data, making API calls, or processing information.

## The Building Blocks: LLMs and Tools

### Large Language Models (LLMs)

LLMs like GPT-4, Claude, or open-source models like Qwen2 are the “brain” of our agent. They excel at:

*   Understanding natural language
*   Generating coherent text
*   Following instructions
*   Reasoning about problems

However, LLMs have limitations:

*   They can’t access real-time data
*   They can’t call APIs
*   They can’t perform calculations beyond what’s in their training

### Tools: Extending LLM Capabilities

Tools are functions that agents can call to extend their capabilities. A tool might:

*   Query a database
*   Call an external API
*   Perform calculations
*   Parse data files
*   Search the web

When you give an agent access to tools, it can:

1.  Analyze the user’s query
2.  Decide which tools to use
3.  Call those tools with appropriate parameters
4.  Use the results to formulate a response

This is called function calling or tool use—one of the most powerful features of modern LLM frameworks.

## Our Project: A Disaster Monitoring Agent

Let’s build an agent that can answer questions about global disasters by:

1.  Fetching real-time disaster data from GDACS (Global Disaster Alert and Coordination System)
2.  Searching for disasters by location or country
3.  Providing detailed information about disaster events

### Architecture Overview

`User Query → Agent → LLM → Tool Selection → Execute Tools → Synthesize Response`

The agent will have three tools:

1.  `get_lat_long`: Get coordinates for a town and state
2.  `get_disaster_data`: Find disasters near specific coordinates
3.  `get_disaster_data_from_country`: Find disasters in a specific country

## Step-by-Step Implementation

### 1. Setting Up the Environment

First, we’ll use LangChain for agent orchestration and Ollama for running a local LLM:

```python
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool

import requests
from bs4 import BeautifulSoup

model = ChatOllama(model="qwen2:7b")
```

**Key Components:**

*   `create_agent`: LangChain’s function to create an agent
*   `ChatOllama`: Interface to run local LLMs via Ollama
*   `@tool`: Decorator that converts Python functions into agent tools

### 2. Data Fetching: Parsing the RSS Feed

Before we can build tools, we need to fetch and parse disaster data. GDACS provides an RSS feed with disaster information:

```python
RSS_FEED_URL = "https://www.gdacs.org/xml/rss.xml"

def extract_text(element, default: str = "") -> str:
    """Extract text from an element, returning default if element is None or empty."""
    if element is None:
        return default
    text = element.get_text(strip=True)
    return text if text else default

def parse_item(item) -> dict:
    """Parse a single RSS item and extract relevant fields."""
    data = {}
    data['title'] = extract_text(item.find('title'))
    data['description'] = extract_text(item.find('description'))
    data['georss_point'] = extract_text(item.find('georss:point'))
    
    # Population field
    population = item.find('gdacs:population')
    if population:
        data['gdacs_population_text'] = extract_text(population)
    else:
        data['gdacs_population_text'] = ""
    
    return data

def fetch_rss_feed():
    """Fetch and parse the RSS feed from GDACS."""
    response = requests.get(RSS_FEED_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'xml')
    items = soup.find_all('item')
    return [parse_item(item) for item in items]
```

What’s happening here:

*   We fetch the RSS feed using `requests`
*   Parse XML with `beautifulsoup` library
*   Extract relevant fields (title, description, location, population)
*   Return a list of parsed disaster events

### 3. Building Our First Tool: Location Lookup

Our agent needs to convert place names to coordinates. We’ll use the Open-Meteo geocoding API:

```python
@tool
def get_lat_long(town: str, state: str) -> tuple[float, float]:
    """
    Get the latitude and longitude of the given town and full name of the state.

    Args:
        town: The town to get the latitude and longitude of
        state: The full name of the state of the town

    Returns:
        A tuple containing the latitude and longitude of the town
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": town}
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    # Filter results to match the state with admin1 field
    results = data.get("results", [])
    matching_result = None
    for result in results:
        if result.get("admin1") == state:
            matching_result = result
            break
    
    if matching_result is None:
        raise ValueError(f"No location found for town '{town}' in state '{state}'")
    
    latitude = matching_result.get("latitude")
    longitude = matching_result.get("longitude")
    
    if latitude is None or longitude is None:
        raise ValueError(f"Latitude or longitude not found in result for '{town}', '{state}'")
    
    return (latitude, longitude)
```

**Key Points:**

*   The `@tool` decorator tells LangChain this function is a tool
*   The docstring is crucial—the LLM uses it to understand when and how to call the tool
*   We filter results by state to ensure accuracy
*   Error handling helps the agent understand when something goes wrong

### 4. Building Tool #2: Location-Based Disaster Search

Now we’ll create a tool that finds disasters near specific coordinates:

```python
@tool
def get_disaster_data(lat: float, long: float) -> dict:
    """
    Get the disaster data for the given latitude and longitude.
    Matches events from the RSS feed where the georss_point is within +/- 4 degrees 
    of the given latitude and longitude.

    Args:
        lat: The latitude of the location
        long: The longitude of the location

    Returns:
        A dictionary containing the matching events and the count of matching events
    """
    events = fetch_rss_feed()
    
    # Filter events where lat and long are within +/- 4 degrees
    matching_events = []
    for event in events:
        georss_point = event.get("georss_point", "")
        
        if georss_point:
            try:
                # Parse georss_point (format: "latitude longitude")
                parts = georss_point.strip().split()
                if len(parts) == 2:
                    georss_lat = float(parts[0])
                    georss_long = float(parts[1])
                    
                    # Check if within +/- 4 degrees
                    if (georss_lat - 4 <= lat <= georss_lat + 4 and 
                        georss_long - 4 <= long <= georss_long + 4):
                        matching_events.append(event)
            except (ValueError, IndexError):
                # Skip invalid georss_point values
                continue
    
    return {"events": matching_events, "count": len(matching_events)}
```

What this does:

*   Fetches the latest disaster data from the RSS feed
*   Parses geographic coordinates from each event
*   Filters events within ±4 degrees of the target location
*   Returns matching events with metadata

### 5. Building Tool #3: Country-Based Search

For simpler queries, we’ll add a tool that searches by country name:

```python
@tool
def get_disaster_data_from_country(country: str) -> list:
    """
    Get the disaster data for the given country.

    Args:
        country: The full name of the country

    Returns:
        A list of strings containing matching events
    """
    events = fetch_rss_feed()
    
    matching_events = []
    country_lower = country.lower()
    for event in events:
        title = event.get('title', '').lower()
        if country_lower in title:
            matching_events.append(
                event['title'] + ": " + event['description'] + " " + event['gdacs_population_text']
            )
    
    return matching_events
```

### 6. Creating the Agent

Now we bring everything together:

```python
agent = create_agent(
    tools=[get_lat_long, get_disaster_data, get_disaster_data_from_country], 
    model=model, 
    system_prompt="You are a helpful assistant that can get the latitude and longitude of a town and state, get the disaster data for a given latitude and longitude, and get the disaster data for a given country."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Are there any disasters in India? Briefly describe the disasters."}]}
)

print(result['messages'][-1].content)
```

<pre>
In India, there have been several forest fires that occurred between December 3rd and December 
19th, 2025. Here are brief descriptions of some:


1.  A forest fire started on 09/12/2025 and was still active until 16/12/2025, affecting 22,002 people.

2.  Another fire occurred on 07/12/2025 that lasted until 19/12/2025; it affected 54,311 people in the area.

3.  A similar incident took place on the same date range (07/12 to 19/12/2025), impacting another 36,280 individuals.

4.  There was a fire on 06/12/2025 that lasted until 19/12/2025, resulting in the displacement of 10,384 people.

5.  Lastly, an earlier fire broke out on 03/12/2025 and was under control by 17/12/2025, with an estimated population impact of 50,181.

</pre>

What happens when you run this (Answers may be different for you depending on time):

*   User asks: “Are there any disasters in India?”
*   Agent analyzes the query and decides to use `get_disaster_data_from_country`
*   Agent calls the tool with `country=”India”`
*   Tool fetches RSS feed, filters for India-related events
*   Agent receives the results and synthesizes a natural language response
*   User gets a comprehensive answer about disasters in India

### How Agents Make Decisions

The magic happens in the agent’s decision-making process:

*   **Query Analysis**: The LLM reads the user’s question
*   **Tool Selection**: Based on the query and available tools, it decides which tool(s) to use
*   **Parameter Extraction**: It extracts the right parameters from the query
*   **Tool Execution**: The tool runs and returns data
*   **Response Synthesis**: The LLM combines tool results into a natural response

For example, if you ask: ”What disasters are near Seattle, Washington?”

The agent might:

1.  Recognize it needs coordinates → calls `get_lat_long(”Seattle”, “Washington”)`
2.  Gets coordinates: `(47.9150338, -122.0876997)`
3.  Calls `get_disaster_data(47.9150338, -122.0876997)` to find nearby disasters
4.  Synthesizes the results into a natural response

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Are there any disasters  near Seattle, Washington?"}]}
)

print(result['messages'][-1].content)
```

<pre>
Near Seattle, Washington, there was a green flood alert in the United States which started on 
September 12, 2025 and lasted until December 17, 2025. This event caused 1 death and displaced 294 
people. The exact location of this event is approximately at coordinates 47.9150338, -122.0876997.
</pre>

## Best Practices for Building Agents

### 1. Write Clear Tool Descriptions

The LLM relies on docstrings to understand tools. Be specific:

```python
@tool
def my_tool(param: str) -> str:
    """
    Clear description of what this tool does.
    
    Args:
        param: What this parameter means
    
    Returns:
        What the return value represents
    """
```

### 2. Handle Errors Gracefully

Tools should raise meaningful errors that help the agent understand what went wrong:

```python
if matching_result is None:
    raise ValueError(f"No location found for town '{town}' in state '{state}'")
```

### 3. Keep Tools Focused

Each tool should do one thing well. This makes it easier for the agent to understand when to use it.

### 4. Provide Good System Prompts

Your system prompt should explain the agent’s role and capabilities:

`system_prompt="You are a helpful assistant that can locate with town and state, get the disaster data for a given location."`

### 5. Test with Real Queries

Try various phrasings to ensure your agent handles different question styles:

*   “Are there disasters in Sudan?”
*   “What’s happening with disasters in India?”
*   “Show me disaster alerts for India”

## Advanced Concepts

### Tool Chaining

Agents can chain multiple tools together. For example:

1.  User: “What disasters are near Boston?”
2.  Agent calls `get_lat_long(”Boston”, “Massachusetts”)`
3.  Agent uses those coordinates to call `get_disaster_data(lat, long)`
4.  Agent synthesizes the final response

The agent automatically figures out this chain based on the query!

### Error Recovery

Good agents can recover from errors:

```python
try:
    result = tool_call()
except ValueError as e:
    # Agent can try alternative approaches
    pass
```

### Streaming Responses

For better UX, you can stream agent responses as they’re generated, giving users real-time feedback.

## Real-World Applications

AI agents with tooling are being used for:

*   **Customer Support**: Agents that can query order databases, check shipping status, process returns
*   **Data Analysis**: Agents that can query databases, generate reports, create visualizations
*   **Content Creation**: Agents that can search the web, fact-check, and cite sources
*   **Software Development**: Agents that can read code, run tests, make changes

## Conclusion

AI agents represent a powerful paradigm shift—from static chatbots to dynamic systems that can interact with the real world. By combining LLMs with tools, we can build assistants that:

*   Access real-time data
*   Perform complex operations
*   Answer questions that require external information
*   Adapt to new situations

The GDACS disaster monitoring agent we built demonstrates these principles in action. You can extend this pattern to build agents for any domain—whether it’s monitoring stock prices, analyzing weather data, or managing your calendar.

## Next Steps

1.  **Experiment**: Try modifying the tools or adding new ones
2.  **Improve**: Add error handling, caching, or rate limiting
3.  **Deploy**: Wrap your agent in a web interface or API
4.  **Scale**: Add more sophisticated tools and capabilities

The future of AI is not just in better models, but in better ways to connect them to the world. Agents with tooling are how we make that connection.

Want to see the full code? Check out the GitHub repository [https://github.com/antosara-dev/disaster-agent](https://github.com/antosara-dev/disaster-agent) and try running it yourself following the `README.md`
