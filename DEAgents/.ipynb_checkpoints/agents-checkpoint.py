# agents/migration_agents.py
from crewai import Agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import vertexai
from langchain_google_vertexai import ChatVertexAI
from tools.schema_listing_tool import list_bigquery_objects
from tools.object_metadata_tool import get_object_metadata
import os
from google.oauth2 import service_account
import vertexai
from langchain_google_vertexai import ChatVertexAI
import litellm
from crewai.llm import LLM
from crewai import Agent, Crew
from tools.object_metadata_tool import get_object_metadata


# --- Init Vertex --- 
# Set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "intelia-health-8e5f53efd3df.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "intelia-health"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
model = ChatVertexAI(model="gemini-2.5-pro" , project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"])
 

schema_extractor_agent = Agent(
    role="Schema Extractor",
    goal="List all bigquery dataset objects (tables, views, procedures, functions)",
    backstory="Expert at reverse-engineering database structures.",
    tools=[list_bigquery_objects],
    llm=model,
    verbose=True
)

object_metadata_agent = Agent(
    role="Object Metadata Retriever",
    goal="Get full metadata (columns, types) of a bigquery object",
    backstory="Knows how to introspect database objects in detail.",
    tools=[get_object_metadata],
    llm=model,
    verbose=True
)

bq_mapper_agent = Agent(
    role="BigQuery Mapper",
    goal="Generate BigQuery-compatible schema and mappings",
    backstory="Cloud migration specialist converting SQL Server to BigQuery.",
    llm=model,
    verbose=True
)
