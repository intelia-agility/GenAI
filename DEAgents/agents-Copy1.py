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


# Set environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "intelia-health-8e5f53efd3df.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "intelia-health"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Load credentials from service account file
credentials = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)

# Initialize Vertex AI with credentials
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials,
)

class ChatVertexAIWithCall(ChatVertexAI):
    def __init__(self, model, project, location, credentials, **kwargs):
        super().__init__(model=model, project=project, location=location, **kwargs)
        self._vertex_project = project
        self._vertex_location = location
        self._vertex_model = model
        self._vertex_credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        print('8888')

    def call(self, messages, **kwargs):
        print('*****you are here***')
        return litellm.completion(
            model=self._vertex_model,
            messages=messages,
            vertex_project=self._vertex_project,
            vertex_location=self._vertex_location,
            vertex_credentials=self._vertex_credentials,
            **kwargs,
        )


llm = ChatVertexAIWithCall(
    model="gemini-2.5-pro",
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
    credentials=credentials,
)

class LLMWrapper:
    def __init__(self, llm_instance):
        self.llm = llm_instance
    
    def generate(self, prompt, **kwargs):
        # The 'prompt' here should be converted into the messages format expected by your llm.call
        messages = [{"role": "user", "content": prompt}]
        # Call your llm.call and get the response
        response = self.llm.call(messages=messages, **kwargs)
        # Assuming response is a dict or object with 'text' or similar key - adjust if different
        # For litellm, the response might be just the text string, so just return it
        return response

llm = LLMWrapper(llm)

schema_extractor_agent = Agent(
    role="Schema Extractor",
    goal="List all bigquery dataset objects (tables, views, procedures, functions)",
    backstory="Expert at reverse-engineering database structures.",
    tools=[list_bigquery_objects],
    llm=llm,
    verbose=True
)

object_metadata_agent = Agent(
    role="Object Metadata Retriever",
    goal="Get full metadata (columns, types) of a bigquery object",
    backstory="Knows how to introspect database objects in detail.",
    tools=[get_object_metadata],
    llm=llm,
    verbose=True
)

bq_mapper_agent = Agent(
    role="BigQuery Mapper",
    goal="Generate BigQuery-compatible schema and mappings",
    backstory="Cloud migration specialist converting SQL Server to BigQuery.",
    llm=llm,
    verbose=True
)
