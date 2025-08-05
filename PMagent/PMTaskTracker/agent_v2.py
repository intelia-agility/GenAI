import os

from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from crewai.tools import tool
from crewai_tools import SerperDevTool

import vertexai
import json
from crewai.project import CrewBase, agent, crew, task



# Load configuration from config.json
with open('./PMTaskTracker/config.json') as config_file:
         config = json.load(config_file)

PROJECT_ID=config['project']
LOCATION=config['location']
serper_api_key=config['serper_api_key']
 
vertexai.init(project=PROJECT_ID, location=LOCATION )
gemini_llm= "vertex_ai/gemini-2.0-flash-001"

os.environ["SERPER_API_KEY"] = serper_api_key  # serper.dev API key

model = "vertex_ai/gemini-2.0-flash"
# Loading Tools
search_tool = SerperDevTool()


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
    def kickoff(self, input: str):          

            researcher = Agent(
            role='Senior Research Analyst',
            goal='Uncover cutting-edge developments in AI and data science',
            backstory=(
                "You are a Senior Research Analyst at a leading tech think tank. "
                "Your expertise lies in identifying emerging trends and technologies in AI and data science. "
                "You have a knack for dissecting complex data and presenting actionable insights."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[search_tool,get_product_details,get_product_price],
            llm=model
            
        )
            research_task = Task(
                description=(
                    "Conduct a comprehensive analysis of the latest advancements in AI in 2025. "
                    "Identify key trends, breakthrough technologies, and potential industry impacts. "
                    "Compile your findings in a detailed report. "
                    "Make sure to check with a human if the draft is good before finalizing your answer."
                ),
                expected_output='A comprehensive full report on the latest AI advancements in 2025, leave nothing out',
                agent=researcher,
                human_input=False
            )


            writer = Agent(
            role='Tech Content Strategist',
            goal='Craft compelling content on tech advancements',
            backstory=(
                "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation. "
                "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
            ),
            verbose=True,
            allow_delegation=True,
            tools=[search_tool],
            cache=False,  # Disable cache for this agent
            llm=model
        )

            writing_task = Task(
                description=(
                    "Using the insights from the researcher\'s report, develop an engaging blog post that highlights the most significant AI advancements. "
                    "Your post should be informative yet accessible, catering to a tech-savvy audience. "
                    "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
                ),
                expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2025',
                agent=writer,
                human_input=False #if you want human input, set this to true
            )

             # # Instantiate your crew with a sequential process
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, writing_task],
                process=Process.sequential,
                verbose=True,
                #memory=True,-->this is only available in openapi
                #planning=True, # Enable planning feature for the crew -->this is only available in openapi
                llm=model
            )

 
        # Get your crew to work!
            result = crew.kickoff()
            return  result 




local_custom_agent = CrewAIApp(project=PROJECT_ID, location=LOCATION)
local_custom_agent.set_up()

response = local_custom_agent.kickoff(input="")
print(response)
