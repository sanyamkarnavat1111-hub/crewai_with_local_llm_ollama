from crewai import LLM 
from crewai import Agent, Task, Crew


llm = LLM(
    model="ollama/llama3:8b",
    base_url="http://localhost:11434",
    **{
        "num_gpu": 20,          # 20 layers on GPU (out of ~32 for 8B)
        "num_thread": 6,        # CPU threads for remaining layers
        "num_ctx": 4096
    }
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
    process="sequential"  # Default
)

# Hierarchical Process - Manager agent delegates and manages other agents
manager_agent = Agent(
    role="Project Manager",
    goal="Ensure all research and writing tasks are completed effectively",
    backstory="You are an experienced project manager who coordinates multiple team members.",
    llm=llm,
    allow_delegation=True
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    manager_agent=manager_agent,  # Manager oversees task delegation
    process="hierarchical"
)


result = crew.kickoff()
print("\nFinal Result:\n", result)