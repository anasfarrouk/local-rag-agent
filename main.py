from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
import asyncio, uvloop, os, logging


# set the local models and the url of the server
llm: str = os.getenv("LLM", "qwen3:0.6b")
embedding: str = os.getenv("EMBEDDING", "qwen3-embedding:0.6b")
url: str = os.getenv("URL", "http://localhost:11434")

logging.disable(logging.CRITICAL)

# Set both LLM and embedding model
Settings.llm = Ollama(
    model=llm,
    base_url=url,
)
Settings.embed_model = OllamaEmbedding(
    model_name=embedding,
    base_url=url,
)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index with Ollama embeddings
index = VectorStoreIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()

# search tool
async def search_documents(query: str) -> str:
    '''
    This function search the given documents for an answer the query parameter.
    It returns the answer it found from the documents.
    '''
    response = await query_engine.aquery(query)
    return str(response)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools = [search_documents],
    llm = Ollama(
        model=llm,
        base_url=url,
    ),
    system_prompt = "You are a helpful assistant. You can use the given tool by calling search_documents and provide the question you are asked as query"
)

# create context
ctx = Context(agent)

# start the conversation
async def main() -> None:
    while True:
        prompt: str = input("\033[32mUser \033[34m>>> \033[0m") 
        if prompt == "/exit":
            break
        response: str = await agent.run(prompt, ctx=ctx)
        print(f"\033[33mAgent \033[34m>>> \033[0m{response}")

if __name__ == "__main__":
    uvloop.run(main())

