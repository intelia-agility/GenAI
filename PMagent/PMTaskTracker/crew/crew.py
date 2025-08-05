from datetime import datetime, timedelta
from typing import List
from crewai.tools import tool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tasks.conditional_task import ConditionalTask
#import crewai.tasks.conditional_task as conditional_task

from crewai.tasks.task_output import TaskOutput

from langchain_google_vertexai import ChatVertexAI
import os
import vertexai
import yaml
from google.auth import default
from datetime import date as Date
from datetime import datetime, timedelta
from typing import List
from google.cloud import storage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from datetime import datetime, timedelta
from typing import List
from crewai.tools import tool
from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
    HarmBlockThreshold,
    HarmCategory,
)

# --- Load project config ---
with open("PMTaskTracker/crew/config/projectconfig.yaml", 'r') as file:
    project_config = yaml.safe_load(file)

PROJECT_ID = project_config['project'].strip()
LOCATION = project_config['location'].strip()
serper_api_key = project_config['serper_api_key'].strip()

# --- Init Vertex ---
vertexai.init(project=PROJECT_ID, location=LOCATION)
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
model = ChatVertexAI(model="gemini-1.5-flash" , project=PROJECT_ID,
    location=LOCATION)

import subprocess
import os

class EventOutput(BaseModel):
    events: dict

def nothing_to_process_condition(output: TaskOutput) -> bool:
    return len(output.pydantic.events) ==0  # this will skip this task

def deduplicate_transcripts(date: str, project_config: dict) -> dict:
        """
        Skips files already processed (present in processed_set).
        """

        print('check for duplicates files...')
     
        bucket_name= project_config['dailyreport_bucket_name'].strip()
        folder=project_config['processed_dailyupdate_folder_prefix'].strip()
        file_name=project_config['processed_dailyupdate_file_prefix'].strip()


        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=f"{folder}{date}"))
        if len(blobs) > 0: 
             return True
        else: 
            return False
        
def clean_transcript_tool(content: str) -> str:
        """
        Cleans a transcript by removing redundant lines, extra whitespace, etc.
        This can be expanded with more logic.
        """
        print('cleaning files...')


        cleaner_llm = GenerativeModel("gemini-1.5-flash")

        CLEANING_PROMPT = f"""
        You are a transcript cleaning assistant. Given a daily standup transcript, remove greetings, redundant lines, filler phrases (like “Good morning team!” or “apologies, I disconnected”), and keep only the meaningful update lines.

        Format the output as a clean list of updates per person, like this:

        Abhi:
        - Finished Module 1 of Collibra training.
        - Working on the LAB today.

        Tara:
        - Working on CrewAI integration with LangGraph.
        - Reviewing AusPost documentation.

        Transcript:
        {content}

        Cleaned Output:
        """      
        response = cleaner_llm.generate_content(CLEANING_PROMPT) 

        return response.text.strip()




@tool
def parse_date_range_tool(date_or_range: str) -> List[str]:
    """
    Parses a single date or date range string into a list of date strings (YYYY-MM-DD).
    """
    date_or_range = date_or_range.strip()
    if "to" in date_or_range:
        start_str, end_str = [d.strip() for d in date_or_range.split("to")]
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start_date).days + 1)]
    else:
        single_date = datetime.strptime(date_or_range, "%Y-%m-%d")
        return [single_date.strftime("%Y-%m-%d")]

@tool
def get_project_config() -> dict:
        """
        gets the project configurations
      
        """
        return project_config
        
@tool
def process_gcs_files_tool(dates: List[str],project_config: dict):
        """
        access, process, and store transcripts in GCS for the given date(s) based on the project configurations
      
        """
        
        print('processing files...')
        print(dates)

        bucket_name= project_config['raw_dailyupdate_bucket_name'].strip()
        folder=project_config['raw_dailyupdate_folder_prefix'].strip()     
        file_format=project_config['raw_dailyupdate_file_format'].strip()
        file_name=project_config['raw_dailyupdate_file_prefix'].strip() 

        target_folder_prefix=project_config['processed_dailyupdate_folder_prefix'].strip()
        target_file_prefix=project_config['processed_dailyupdate_file_prefix'].strip()
        
       
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        all_files = {}

    
        for date in dates:
            dt = datetime.strptime(date, "%Y-%m-%d").strftime("%d%m%Y") 
            blobs = bucket.list_blobs(prefix=f"{folder}{str(dt)}")

            #deduplicate files
            if deduplicate_transcripts(str(dt),project_config):
                    continue
            #process, clean and store files in gcs
            for blob in blobs:
                if not (blob.name.endswith(file_format) and blob.name.split('/')[-1].startswith(file_name)):
                    continue
          
                file_content = blob.download_as_text()                     
                content=clean_transcript_tool(file_content)       
                target_name = blob.name.replace(f"{folder}", target_folder_prefix).replace(f"{file_name}",target_file_prefix)
                blob = bucket.blob(target_name)
                blob.upload_from_string(content) 
        
@tool
def list_processed_files_tool(dates: List[str],project_config: dict)-> dict:
        """
        access, and retrieve the content of transcripts for the given date(s) from GCS based on the project configurations  
      
        """
        
        print('retrieving data files...')

        bucket_name= project_config['raw_dailyupdate_bucket_name'].strip()
        folder=project_config['processed_dailyupdate_folder_prefix'].strip()
        file=project_config['processed_dailyupdate_file_prefix'].strip()
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        all_files = {}

    
        for date in dates:
            dt = datetime.strptime(date, "%Y-%m-%d").strftime("%d%m%Y") 
            blobs = bucket.list_blobs(prefix=f"{folder}{str(dt)}")
            
            #process, clean and store files in gcs
            for blob in blobs:   
                if  (blob.name.split('/')[-1].startswith(file)):
                    content = blob.download_as_text()                                 
                    all_files[blob.name] = content

        return all_files

@tool
def send_email_tool(subject: str, body: str):
    """
    Send an email to the project manger with the given the analysed result in body and email subject
    """
    import smtplib
    from email.message import EmailMessage

    print('sending email...')

    sender_email = "tara.pourhabibi@intelia.com.au"
    sender_password = "aaqj sbxf hmev omtt"  # Use an app password or secret

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = 'tara.pourhabibi@intelia.com.au'

    print('sending email........')
    print(body)
    print(subject)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)
        
# @tool
# def send_email_tool(subject: str, body: str):
#     """
#     Send an email to the project manger with the given the analysed result in body and email subject
#     """
#     import smtplib
#     from email.message import EmailMessage

#     print('sending email...')

#     sender_email = "tahereh.pourhaibi@gmail.com"
#     sender_password = "aaqj sbxf hmev omtt"  # Use an app password or secret

#     msg = EmailMessage()
#     msg.set_content(body)
#     msg["Subject"] = subject
#     msg["From"] = sender_email
#     msg["To"] = 'tara.pourhabibi@intelia.com.au, direnc.uysal@intelia.com.au'

#     import requests

#     tenant_id = "1c490c50-5eeb-431f-9afe-21f8cf77048b"
#     client_id = "3312d438-dd5b-4f7e-b941-21a4f3bc5e6a"
#     client_secret = "nbI8Q~UiJdL.k_WN-BbFli9gj2bahk2QxRRxedqu"
#     scope = "https://graph.microsoft.com/.default"

#     token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
#     data = {
#         'grant_type': 'client_credentials',
#         'client_id': client_id,
#         'client_secret': client_secret,
#         'scope': scope
#     }

#     token_response = requests.post(token_url, data=data)
#     access_token = token_response.json().get("access_token")

#     email_url = "https://graph.microsoft.com/v1.0/users/tahereh.pourhabibi@gmail.com/sendMail"

#     headers = {
#         "Authorization": f"Bearer {access_token}",
#         "Content-Type": "application/json"
#     }

#     email_msg = {
#         "message": {
#             "subject": subject,
#             "body": {
#                 "contentType": "Text",
#                 "content": body
#             },
#             "toRecipients": [
#                 {"emailAddress": {"address": "tahereh.pourhabibi@gmail.com"}},
#                 {"emailAddress": {"address": "tara.pourhabibi@intelia.com.au"}}
#             ]
#         }
#     }

#     response = requests.post(email_url, headers=headers, json=email_msg)

#     if response.status_code == 202:
#         print("Email sent successfully!")
#     else:
#         print(f"Failed to send email: {response.status_code}, {response.text}")


@CrewBase
class CrewaiProjectManagerTaskTrackingagent:
    """CrewAI Project Manager Task Tracking Agent"""

    def __init__(self):
        # Load configs into instance variables
        with open("PMTaskTracker/crew/config/agents.yaml", 'r') as f:
            self.agents_config = yaml.safe_load(f)

        with open("PMTaskTracker/crew/config/tasks.yaml", 'r') as f:
            self.tasks_config = yaml.safe_load(f)
    #--
    @agent
    def User_input_collector_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["User_input_collector_agent"],
            llm=model,
            verbose=False,
            human_input=True
        )

    @task
    def date_collector_task(self) -> Task:
        return Task(
            config=self.tasks_config["date_collector_task"],
            agent=self.User_input_collector_agent()
        )

    #---
    @agent
    def Transcript_collector_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Transcript_collector_agent"],
            llm=model,
            verbose=False,
            human_input=False,
            tools=[parse_date_range_tool, get_project_config,parse_date_range_tool,process_gcs_files_tool, list_processed_files_tool],
            
        )

    @task
    def transcript_collection_task(self) -> Task:
        return Task(
            config=self.tasks_config["transcript_collection_task"],
            agent=self.Transcript_collector_agent() ,
            output_pydantic=EventOutput,
            
        )

    
    ###
     #---
    @agent
    def Data_validator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Data_validator_agent"],
            llm=model,
            verbose=False,
            human_input=False,            
        )

    @task 
    def data_validation_task(self) -> ConditionalTask:
        ct= ConditionalTask(
        agent=self.Data_validator_agent(),
        config=self.tasks_config["data_validation_task"],
        condition=nothing_to_process_condition)      
        return ct

 #---
    @agent
    def Standup_summarizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["Standup_summarizer_agent"],
            llm=model,
            verbose=False,
            human_input=False    ,
            tools=[send_email_tool]          
        )

    @task
    def report_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["report_generation_task"],
            agent=self.Standup_summarizer_agent() 
        
            
        )
    # #---
    # @agent
    # def Report_Generator_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["Report_Generator_agent"],
    #         llm=model,
    #         verbose=False,
    #         human_input=False,
	# 		tool=[send_email_tool]
             
    #     )

    # @task
    # def report_generation_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["report_generation_task"],
    #         agent=self.Report_Generator_agent() 
        
            
    #     )
      
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=0,
            llm=model
        )
