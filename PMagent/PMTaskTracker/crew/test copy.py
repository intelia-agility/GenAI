from datetime import datetime, timedelta
from typing import List
from crewai.tools import tool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
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

dates=['2025-04-17']

def deduplicate_transcripts(date: Date, project_config: dict) -> dict:
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

        CLEANING_PROMPT = """
        You are a transcript cleaning assistant. Given a daily standup transcript, remove greetings, redundant lines, filler phrases (like “Good morning team!” or “apologies, I disconnected”), and keep only the meaningful update lines.

        Format the output as a clean list of updates per person, like this:

        Abhi:
        - Finished Module 1 of Collibra training.
        - Working on the LAB today.

        Tara:
        - Working on CrewAI integration with LangGraph.
        - Reviewing AusPost documentation.

        Transcript:
        {transcript}

        Cleaned Output:
        """
        response = cleaner_llm.generate_content(CLEANING_PROMPT.format(transcript=content)) 
                        
                         
        return response.text.strip()



# if 1==1:
      
#         print('processing files...')
#         print(dates)
  
#         bucket_name= project_config['raw_dailyupdate_bucket_name'].strip()
#         folder=project_config['raw_dailyupdate_folder_prefix'].strip()     
#         file_format=project_config['raw_dailyupdate_file_format'].strip()
#         file_name=project_config['raw_dailyupdate_file_prefix'].strip() 

#         target_folder_prefix=project_config['processed_dailyupdate_folder_prefix'].strip()
#         target_file_prefix=project_config['processed_dailyupdate_file_prefix'].strip()

#         client = storage.Client()
#         bucket = client.bucket(bucket_name)
#         all_files = {}
    
#         for date in dates:
#             dt = datetime.strptime(date, "%Y-%m-%d").strftime("%d%m%Y") 
#             blobs = bucket.list_blobs(prefix=f"{folder}{str(dt)}")

#             if deduplicate_transcripts(date,project_config):
#                 continue
       
#             for blob in blobs:
#                 if not (blob.name.endswith(file_format) and blob.name.split('/')[-1].startswith(file_name)):
#                     continue
#                 content = blob.download_as_text()
#                 file_content = blob.download_as_text()
#                 content=clean_transcript_tool(file_content)                
#                 target_name = blob.name.replace(f"{folder}", target_folder_prefix).replace(f"{file_name}",target_file_prefix)
#                 all_files[blob.name] = content
#                 blob = bucket.blob(target_name)
#                 blob.upload_from_string(content) 
#         print(all_files)

import subprocess
import os
import  zoom_meeting_sdk as zoom

def join_meeting(MEETING_ID,MEETING_PWD):
        mid = MEETING_ID
        password = MEETING_PWD
        display_name = "My meeting bot"

        meeting_number = int(mid)

        join_param = zoom.JoinParam()
        join_param.userType = zoom.SDKUserType.SDK_UT_WITHOUT_LOGIN

        param = join_param.param
        param.meetingNumber = meeting_number
        param.userName = display_name
        param.psw = password
        param.isVideoOff = False
        param.isAudioOff = False
        param.isAudioRawDataStereo = False
        param.isMyVoiceInMix = False
        param.eAudioRawdataSamplingRate = zoom.AudioRawdataSamplingRate.AudioRawdataSamplingRate_32K

        join_result = self.meeting_service.Join(join_param)
        print("join_result =",join_result)

        self.audio_settings = self.setting_service.GetAudioSettings()
        self.audio_settings.EnableAutoJoinAudio(True)

def record_zoom_meeting_tool(meeting_id, passcode, topic, gcs_bucket, gcs_path):
    # Initialize the Zoom Meeting SDK
    sdk = ZoomMeetingSDK(
        client_id='fhoqlBo1QFyEk8kU7Wwyhg',
        client_secret='zs8olDayzNYGmgMBHcDGBZnYQvEuDZMP'
    )

    # Join the Zoom meeting

    meeting = sdk.join_meeting(meeting_id=meeting_id, passcode=passcode)

    # Start recording
    recording = meeting.start_recording()

    # Wait for the meeting to end
    meeting.wait_until_end()

    # Stop recording
    recording.stop()

    # Retrieve the path to the recorded file
    recording_file_path = recording.get_file_path()

    # Upload the recording to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(recording_file_path)

    # Optional: Remove the local recording file
    os.remove(recording_file_path)

##print(record_zoom_meeting_tool('78679564914', '3BSH0S', 30,"pmtask_tracker","test") )