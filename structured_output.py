from crewai import LLM
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import List
import json
from typing import Optional

class BlogPost(BaseModel):
    """Structured output for a blog post."""
    title: str  # Required: Always provide a value.
    content: str  # Required.
    tags: Optional[list[str]] = None  # Optional: Defaults to None if missing.
    word_count: Optional[int] = None  # Optional.

# === 2. Setup LLM ===
llm = LLM(
    # model="ollama/llama3:8b",
    model="ollama/gemma2:9b",
    base_url="http://localhost:11434"
)

# === 3. Agents ===
blog_agent = Agent(
    role="Senior Blog Content Writer",
    goal="Generate engaging blog posts on given topics",
    backstory="""You are a professional writer with 10+ years in journalism.
    Always structure your output as a JSON object with fields: title (string), 
    content (string, under 200 words), tags (array of strings), and word_count (integer).""",
    verbose=True,
    allow_delegation=False,
    llm=llm  
)


generate_blog_task = Task(
    description="""Write a blog post on {topic}. 
    Output ONLY a JSON object with the following exact fields:
    - title: A catchy string title.
    - content: The full post body as a string (keep under 200 words).
    - tags: An array of 3-5 relevant strings.
    - word_count: An integer count of words in the content.""",
    expected_output="A JSON object conforming to the BlogPost schema.",
    agent=blog_agent,
    output_pydantic=BlogPost,  # Enforces Pydantic parsing.
    # Alternative: Use output_json=BlogPost for raw JSON output (still parseable to Pydantic).
)



# === 6. Run Crew ===
crew = Crew(
    agents=[blog_agent],
    tasks=[generate_blog_task],
    verbose=True  # High verbosity to inspect LLM calls if issues arise.
)

result = crew.kickoff(inputs={
    "topic" : "Machine Learning"
})
structured_output = result.pydantic  # Full Pydantic instance

print("===== Title =====")
print(structured_output.title)
print("-"*100)


print("===== Content =====")
print(structured_output.content)
print("-"*100)

print("===== Tags =====")
print(structured_output.tags)
print("-"*100)


print("===== Word Count =====")
print(structured_output.word_count)
print("-"*100)
