from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

llm = ChatZhipuAI(
    temperature=0.5,
    model="glm-4-flash"  # free-quota model (glm-5.2 needs account balance, error 1113)
)

# Schema for structured output
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

print("=== output (structured) ===")
print(output.model_dump())

# Define a tool
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")

print("\n=== msg (with tools) ===")
msg.pretty_print()

# Get the tool call
msg.tool_calls