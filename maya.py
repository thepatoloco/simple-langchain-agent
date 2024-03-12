from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.milvus import Milvus

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Load .env variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Load vector_db
vector_db = Milvus(
    embedding_function=OpenAIEmbeddings(),
    connection_args={ "host": "localhost", "port": "19530" },
    collection_name="banorte_test"
)

# Declarate tools
# Banorte Search
class SearchBanorteInput(BaseModel):
    query: str = Field(description="Search query for vector db with banorte docs")
    
def search_banorte(query: str) -> list[str]:
    """Search in a vector db with banorte info
    Args:
        query (str): search query
    Returns:
        list[str]: most similar results
    """
    return list(map(lambda x: x.page_content, vector_db.similarity_search(query, k=7)))

search_banorte_tool = StructuredTool.from_function(
    func=search_banorte,
    name="search_banorte",
    description="Search in a vector db with banorte info",
    args_schema=SearchBanorteInput,
    handle_tool_error=True
)


tools = [search_banorte_tool]

# Declare llm
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Declare prompt template
system_prompt = '''Tu eres Maya, un asistente virtual de Banorte (el banco mas fuerte de México) que ayuda a los usuarios con su vida financiera.
Por el momento Maya solamente puedes dar información sobre los productos y servicios que ofrece Banorte, dar consejos para mejorar la vida financiera de los clientes.
Maya tiene una personalidad amable, analista, y humana.
Cada vez que Maya menciona un producto de Banorte siempre dices sus ventajas y una desventaja, esto para que paresca un asesor financiero mas realista en vez de una simple promoción de Banorte.
Maya SOLO puede hablar en ESPAÑOL.
Si el cliente habla en un idioma que Maya no maneja, Maya enviara un mensaje de disculpa en español, y le va a mensionar los idiomas en los que si puede hablar.

Hay algunas cosas que Maya no puede hacer por si misma, pero puede utilizar herramientas para hacerlo:
-Recabar información de los productos de Banorte

# Herramientas:
## search_banorte
// Vas a buscar información sobre los productos y servicios de banorte.
// La query de busqueda debe de ser AMPLIA para asegurarte de encontrar la información correcta.
// SIEMPRE debes de buscar la información con esta herramienta antes de responder al cliente.'''
prompt_template = ChatPromptTemplate(
    input_variables=['chat_history', 'input', 'agent_scratchpad'],
    messages=[
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template=system_prompt)),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

# Declare agent
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Declare agent executor
maya_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
