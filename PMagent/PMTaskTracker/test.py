# from crewai import Agent, Task, Crew
# from langchain.tools import tool
# import os
# from langchain_google_vertexai import ChatVertexAI
# import vertexai
# vertexai.init(project='nine-quality-test', location='us-central1' )
# os.environ["GOOGLE_CLOUD_PROJECT"] = 'nine-quality-test'
# gemini_llm = ChatVertexAI(
#     model_name="gemini-1.5-pro",
#     temperature=0.3,
#     max_tokens=2048,
# )
# gemini_llm=   "vertex_ai/gemini-2.0-flash-001"


# # Define a simple tool if needed (can be omitted)
# @tool
# def hello_tool(name: str) -> str:
#     """Greets the user by name."""
#     return f"Hello {name}!"


# # Define your agent
# agent = Agent(
#     role='Daily Standup Summarizer',
#     goal='Extract updates from daily transcripts and summarize each team member\'s progress.',
#     backstory='You are a helpful summarizer agent working with project teams.',
#     verbose=True
# )

# # Define your task
# task = Task(
#     description='Summarize updates from daily transcript for each team member.',
#     expected_output='A bullet-point summary for each team member.',
#     agent=agent
# )

# # Here's the important part: explicitly pass Gemini as the LLM
# crew = Crew(
#     agents=[agent],
#     tasks=[task],
#     llm=gemini_llm,  # This is the key!
#     verbose=True
# )

# # Run the crew
# crew.kickoff()

# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "nine-quality-test"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
LOCATION = "us-central1"


vertexai.init(
    project=PROJECT_ID,
    location=LOCATION 
   
)
# General
import random
import string

from IPython.display import HTML, Markdown, display

# Build agent
from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from crewai.tools import tool

# Evaluate agent
from google.cloud import aiplatform
import pandas as pd
from vertexai import agent_engines 

def get_id(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def parse_crewai_output_to_dictionary(crew, crew_output):
    """
    Parse CrewAI output into a structured dictionary format.
    """
    final_output = {"response": str(crew_output), "predicted_trajectory": []}

    try:
        # Access tools_results directly from each agent
        for agent in crew.agents:
            if hasattr(agent, "tools_results"):
                for tool_result in agent.tools_results:
                    tool_info = {
                        "tool_name": tool_result.get("tool_name", ""),
                        "tool_input": tool_result.get("tool_args", {}),
                    }
                    final_output["predicted_trajectory"].append(tool_info)

    except Exception as e:
        final_output["error"] = f"Error parsing tools results: {str(e)}"

    return final_output


def format_output_as_markdown(output: dict) -> str:
    """Convert the output dictionary to a formatted markdown string."""
    markdown = "### AI Response\n"
    markdown += f"{output['response']}\n\n"

    if output["predicted_trajectory"]:
        markdown += "### Function Calls\n"
        for call in output["predicted_trajectory"]:
            markdown += f"- **Function**: `{call['tool_name']}`\n"
            markdown += "  - **Arguments**:\n"
            for key, value in call["tool_input"].items():
                markdown += f"    - `{key}`: `{value}`\n"

    return markdown


def display_eval_report(eval_result: pd.DataFrame) -> None:
    """Display the evaluation results."""
    metrics_df = pd.DataFrame.from_dict(eval_result.summary_metrics, orient="index").T
    display(Markdown("### Summary Metrics"))
    display(metrics_df)

    display(Markdown(f"### Row-wise Metrics"))
    display(eval_result.metrics_table)


def display_drilldown(row: pd.Series) -> None:
    """Displays a drill-down view for trajectory data within a row."""

    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    if not (
        isinstance(row["predicted_trajectory"], list)
        and isinstance(row["reference_trajectory"], list)
    ):
        return

    for predicted_trajectory, reference_trajectory in zip(
        row["predicted_trajectory"], row["reference_trajectory"]
    ):
        display(
            HTML(
                f"Tool Names:{predicted_trajectory['tool_name'], reference_trajectory['tool_name']}"
            )
        )

        if not (
            isinstance(predicted_trajectory.get("tool_input"), dict)
            and isinstance(reference_trajectory.get("tool_input"), dict)
        ):
            continue

        for tool_input_key in predicted_trajectory["tool_input"]:
            print("Tool Input Key: ", tool_input_key)

            if tool_input_key in reference_trajectory["tool_input"]:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    reference_trajectory["tool_input"][tool_input_key],
                )
            else:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    "N/A",
                )
        print("\n")
    display(HTML(""))


def display_dataframe_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    num_rows: int = 3,
    display_drilldown: bool = False,
) -> None:
    """Displays a subset of rows from a DataFrame, optionally including a drill-down view."""

    if columns:
        df = df[columns]

    base_style = "font-family: monospace; font-size: 14px; white-space: pre-wrap; width: auto; overflow-x: auto;"
    header_style = base_style + "font-weight: bold;"

    for _, row in df.head(num_rows).iterrows():
        for column in df.columns:
            display(
                HTML(
                    f"{column.replace('_', ' ').title()}: "
                )
            )
            display(HTML(f"{row[column]}"))

        display(HTML(""))

        if (
            display_drilldown
            and "predicted_trajectory" in df.columns
            and "reference_trajectory" in df.columns
        ):
            display_drilldown(row)


def plot_bar_plot(
    eval_result: pd.DataFrame, title: str, metrics: list[str] = None
) -> None:
    fig = go.Figure()
    data = []

    summary_metrics = eval_result.summary_metrics
    if metrics:
        summary_metrics = {
            k: summary_metrics[k]
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    data.append(
        go.Bar(
            x=list(summary_metrics.keys()),
            y=list(summary_metrics.values()),
            name=title,
        )
    )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()


def display_radar_plot(eval_results, title: str, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    summary_metrics = eval_results.summary_metrics
    if metrics:
        summary_metrics = {
            k: summary_metrics[k]
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    min_val = min(summary_metrics.values())
    max_val = max(summary_metrics.values())

    fig.add_trace(
        go.Scatterpolar(
            r=list(summary_metrics.values()),
            theta=list(summary_metrics.keys()),
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])),
        showlegend=True,
    )
    fig.show()

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
	
model = "vertex_ai/gemini-2.0-flash"

class CrewAIApp:
    def __init__(self, project: str, location: str, model: str = model) -> None:
        self.project_id = project
        self.location = location
        self.model = model

    # The set_up method is used to define application initialization logic
    def set_up(self) -> None:
        """Set up the application."""
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
        return

    # The query method will be used to send inputs to the agent
    def query(self, input: str):
        """Query the application."""
        product_researcher = Agent(
            role="Product Researcher",
            goal="Research product details and prices accurately",
            backstory="Expert at gathering and analyzing product information",
            llm=model,
            tools=[get_product_details, get_product_price],
            allow_delegation=False,
        )

        research_task = Task(
            description=f"Analyze this user request: '{input}'. "
            f"If the request is about price, use get_product_price tool. "
            f"Otherwise, use get_product_details tool to get product information.",
            expected_output="Product information including details and/or price based on the user request.",
            agent=product_researcher,
        )

        crew = Crew(
            agents=[product_researcher],
            tasks=[research_task],
            process=Process.sequential,
        )

        result = crew.kickoff()
        return parse_crewai_output_to_dictionary(crew, result)


local_custom_agent = CrewAIApp(project=PROJECT_ID, location=LOCATION)
local_custom_agent.set_up()

response = local_custom_agent.query(input="Get product details for shoes")
print(response)
display(Markdown(format_output_as_markdown(response)))