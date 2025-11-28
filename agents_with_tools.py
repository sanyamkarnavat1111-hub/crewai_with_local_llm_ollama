from langchain_community.tools import DuckDuckGoSearchRun
from crewai import LLM, Agent, Task, Crew
from pydantic import BaseModel , Field
from typing import Optional
from crewai.tools import BaseTool
from typing import Type


class SearchInput(BaseModel):
    """Input schema for MyCustomTool."""
    query: str = Field(..., description="User query to be searched on internet")



class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "A tool for searching the web to find current information, news, trends, and recent developments. Use this tool when you need up-to-date information that may not be in your training data."
    


    def _run(self, query: str) -> str:
        """Execute a web search using DuckDuckGo."""
        try:
            search = DuckDuckGoSearchRun()
            results = search.run(query)
            return f"Search results for '{query}': {results}"
        except Exception as e:
            return f"Error occurred while searching for '{query}': {str(e)}"

# Create the search tool instance
search_tool = DuckDuckGoSearchTool()

# Define the output structure
class BlogPost(BaseModel):
    title: str
    content: str
    tags: Optional[list[str]] = None
    word_count: Optional[int] = None

# Setup LLM
llm = LLM(
    model="ollama/llama3:8b",
    # model="ollama/gemma2:9b",
    # model="ollama/dolphin3", # Dolphin 3 is the instruct version of LLama 3 specifically designed to follow instruction , tool calling and agentic ai
    base_url="http://localhost:11434"
)


# Create the agent with the search tool
blog_agent = Agent(
    role="Senior Blog Content Writer",
    goal="Generate engaging, fact-checked blog posts on given topics using web search when necessary",
    backstory="""You are a professional writer with extensive experience in journalism. 
    When working with topics that require current information, recent events, or time-sensitive data, 
    you should use the available DuckDuckGo Search tool to gather up-to-date information before writing.
    
    Always ensure your final output is structured as a complete blog post with a clear title, 
    well-written content, relevant tags, and accurate word count.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]
)

generate_blog_task = Task(
    description="""Write a comprehensive blog post on {topic}. 
    
    Since this topic involves current news and events, first use the DuckDuckGo Search tool to gather 
    the most recent and relevant information. Perform a targeted search to identify the key current 
    developments related to the topic.
    
    After gathering the necessary information, write a blog post that summarizes the most important 
    points. Structure your output as a JSON object containing:
    - title: A compelling and descriptive title for the blog post
    - content: The main body of the post summarizing the key information (keep under 200 words)
    - tags: An array of 3-5 relevant keywords or tags
    - word_count: The approximate number of words in the content""",
    expected_output="A JSON object containing a complete blog post with title, content, tags, and word count",
    agent=blog_agent,
    output_pydantic=BlogPost
)

# Create and execute the crew
crew = Crew(
    agents=[blog_agent],
    tasks=[generate_blog_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "Top 10 cryptocurrency related news for today"})

# Access the structured output
structured_output = result.pydantic

print("Title:", structured_output.title)
print("\nContent:", structured_output.content)
print("\nTags:", structured_output.tags)
print("\nWord Count:", structured_output.word_count)