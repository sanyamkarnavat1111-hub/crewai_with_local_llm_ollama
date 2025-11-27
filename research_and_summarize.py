from crewai import LLM 
from crewai import Agent, Task, Crew


llm = LLM(
    model="ollama/llama3:8b",
    base_url="http://localhost:11434"
)

researcher = Agent(
    role="Researcher",
    goal="Find fun facts about space",
    backstory="You are great at researching scientific facts.",
    llm=llm,
    allow_delegation=False,
)

writer = Agent(
    role="Writer",
    goal="Summarize space facts in a fun way",
    backstory="You are a creative writer who loves to make science engaging.",
    llm=llm,
    allow_delegation=False,
)

# Researcher task
research_task = Task(
    description="Find 3 interesting facts about space that are suitable for kids.",
    expected_output="A bullet list of 3 fun facts about space.",
    agent=researcher,
)

# Writer task
write_task = Task(
    description="Take the space facts and write a short fun paragraph summarizing them.",
    expected_output="A short paragraph in a fun tone summarizing the facts.",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True,  ### ensures you’ll see detailed logs of what’s happening during execution.
)


result = crew.kickoff()
print("\nFinal Result:\n", result)