

import os
import vertexai   
# General
import random
import string
import warnings

from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Build agent
from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from crewai.tools import tool

# Evaluate agent
from google.cloud import aiplatform
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import (
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    TrajectorySingleToolUse,
)


PROJECT_ID = "nine-quality-test"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION, experiment=EXPERIMENT_NAME)
model = "vertex_ai/gemini-1.5-pro-002"

def parse_crewai_output_to_dictionary(crew, crew_output):
    """
    Parse CrewAI output into a structured dictionary format.
    """
    final_output = {"response": str(crew_output), "predicted_trajectory": []}

    for agent in crew.agents:
        try:
            for tool_result in agent.tools_results:
                tool_info = {
                    "tool_name": tool_result.get("tool_name", ""),
                    "tool_input": tool_result.get("tool_args", {}),
                }
                final_output["predicted_trajectory"].append(tool_info)
        except AttributeError as e:
            final_output["error"] = f"Agent does not have tools_results: {str(e)}"
            print(f"Error: {e}")

@tool
def get_product_details(product_name: str):
    """Gathers basic details about a product."""
    details = {
        "smartphone": "A cutting-edge smartphone with advanced camera features and lightning-fast processing.",
        "usb charger": "A super fast and light usb charger",
        "shoes": "High-performance running shoes designed for comfort, support, and speed.",
        "headphones": "Wireless headphones with advanced noise cancellation technology for immersive audio.",
        "speaker": "A voice-controlled smart speaker that plays music, sets alarms, and controls smart home devices.",
    }
    return details.get(product_name, "Product details not found.")


@tool
def get_product_price(product_name: str):
    """Gathers price about a product."""
    details = {
        "smartphone": 500,
        "usb charger": 10,
        "shoes": 100,
        "headphones": 50,
        "speaker": 80,
    }
    return details.get(product_name, "Product price not found.")
  

class ProductFlow(Flow):
    @start
    def begin_flow(self):
        """Starts the product information flow"""
        return "check_request"

    @listen("check_request")
    def router(self, state: dict) -> str:
        """Routes the product request to appropriate handler"""
        # Get the last message from the state
        last_message = state.get("last_message", {})
        tool_calls = last_message.get("tool_calls", [])

        if tool_calls:
            function_name = tool_calls[0].get("name")
            if function_name == "get_product_price":
                return "get_product_price"
            else:
                return "get_product_details"
        return "end"


def agent_parsed_outcome(input):
    product_researcher = Agent(
        role="Product Researcher",
        goal="Research product details and prices accurately",
        backstory="Expert at gathering and analyzing product information",
        llm=model,
        tools=[get_product_details, get_product_price],
        allow_delegation=False,
    )

    # Create task based on the input
    research_task = Task(
        description=f"Analyze this user request: '{input}'. "
        f"If the request is about price, use get_product_price tool. "
        f"Otherwise, use get_product_details tool to get product information.",
        expected_output="Product information including details and/or price based on the user request.",
        agent=product_researcher,
    )

    # Create crew with sequential process
    crew = Crew(
        agents=[product_researcher],
        tasks=[research_task],
        process=Process.sequential,
    )

    result = crew.kickoff()
    return parse_crewai_output_to_dictionary(crew, result)