from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool
from pydantic import BaseModel
from typing import Optional

# Define the output structure
class SummaryReport(BaseModel):
    summary: str
    key_points: list[str]

# Explicitly create an LLM instance
llm = LLM(
    model="ollama/llama3:8b",  # Specify the model you want to use
    base_url="http://localhost:11434"  # Ollama server URL
)

# Create the tool
read_tool = FileReadTool()

# Create an agent and explicitly assign the LLM
research_agent = Agent(
    role="Research Assistant",
    goal="Read information from files and create summaries",
    backstory="You are a research assistant whose job is to read content from files and create clear, concise summaries of that content.",
    verbose=True,
    allow_delegation=False,
    llm=llm,  # Explicitly specify the LLM here
    tools=[read_tool]
)

# Create the task
research_task = Task(
    description="""
    Read the content from the file named {file_name} using the File Read tool.
    
    After reading the content, create a clear and concise summary of what you find in the file.
    
    Your final output should be a structured summary of the content from the file.
    """,
    expected_output="A concise summary of the content read from the file",
    agent=research_agent,
    output_pydantic=SummaryReport
)

# Create and execute the crew
crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

# Execute the task
result = crew.kickoff(inputs={
    "file_name" : "quick_guide.txt"
})

# Print the result
print("Summary Report:")
print(f"Summary: {result.pydantic.summary}")
print(f"Key Points: {result.pydantic.key_points}")