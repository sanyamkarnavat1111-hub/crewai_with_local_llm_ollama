import sqlite3
from crewai.tools import BaseTool
from crewai import Agent , Task , LLM , Crew

class SQLiteQueryTool(BaseTool):
    name: str = "SQLite Query Tool"
    description: str = "Executes a given SQL query against the SQLite database and returns the results."

    def _run(self, query: str) -> str:
        conn = None
        try:
            conn = sqlite3.connect('Users.db')
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.commit() # Commit changes for INSERT, UPDATE, DELETE
            return str(results)
        except sqlite3.Error as e:
            return f"Database error: {e}"
        finally:
            if conn:
                conn.close()



# Create an instance of your custom tool
custom_sqlite_tool = SQLiteQueryTool()

# Explicitly create an LLM instance
llm = LLM(
    model="ollama/llama3:8b",  # Specify the model you want to use
    base_url="http://localhost:11434"  # Ollama server URL
)


# Create an agent and explicitly assign the LLM
db_agent = Agent(
    role="Database searcher",
    goal="To retrieve the information from database which is relevant to user query",
    backstory="You are a Sqlite database searches which as full capacity to generate SQL query and execute them and your job is to understand user query provided and retrieve most relevant information from database and present them.",
    verbose=True,
    allow_delegation=False,
    llm=llm,  # Explicitly specify the LLM here
    tools=[custom_sqlite_tool]
)


db_search_task = Task(
    description="""
    Understand the following User query :- " {user_query} "  and use SQLiteQueryTool to generate and execute SQLite queries to retrieve information from database,
    which can help answer the user query.
    
    After reading the content, create a clear and concise summary of what you find in the file.
    
    Your final output should be a structured summary of the content from the file.
    """,
    expected_output="A concise summary of the data retrieved from database.",
    agent=db_agent,
)


crew = Crew(
    agents=[db_agent],
    tasks=[db_search_task],
    verbose=True
)

result = crew.kickoff(inputs={
    "user_query" : "Give me all details of candidates who are shortlisted ?"
})


print(result)






