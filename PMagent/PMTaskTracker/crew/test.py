import zoom_meeting_sdk as zoom
import sys,datetime, time, json
import jwt
from dotenv import load_dotenv
import os
import time
import random
import urllib.parse

import gi
gi.require_version('GLib', '2.0')
from gi.repository import GLib

import signal
import sys

testBot = None
main_loop = None
import jwt
import time
import os



def generate_jwt(client_id, client_secret):
    import time

    key = client_id
    secret = client_secret

    iat = int(time.time())
    exp = int((datetime.datetime.today() + datetime.timedelta(days=2)).strftime("%s"))
    tokenExp = int((datetime.datetime.today() + datetime.timedelta(hours=2)).strftime("%s"))

    payload = {
        'appKey': key,
        'iat': iat,
        'exp': exp,
        'tokenExp': tokenExp
    }


    # Your Zoom Meeting Number
    meeting_number = "77390581512"
    # print(f"iat (Issued At): {iat}")
    # print(f"exp (Expiration Time): {exp}")
    # print(f"tokenExp (Token Expiration Time): {exp}")

    # # Create the payload with valid times
    # payload = {
    #     "appKey": "fhoqlBo1QFyEk8kU7Wwyhg",  # Your SDK App Key
    #     "mn": meeting_number,                # Your meeting number
    #     "role": 1,                           # 1 for host, 0 for attendee
    #     "iat": iat-5,                          # Current time
    #     "exp": exp,                          # Expiration time (2 hours)
    #     "tokenExp": exp                       # Token expiration (same as exp)
    # }

    # Your SDK Secret
    secret = "zs8olDayzNYGmgMBHcDGBZnYQvEuDZMP"  # Replace with your actual SDK Secret
    print(payload)

    # Generate JWT
    token = jwt.encode(payload, secret, algorithm="HS256")

    # Print the JWT token
    print(token)

    return token

def get_random_meeting():
    #urls = os.environ.get('MEETING_URLS')
    #url_list = urls.split('\n')
    url_list=['https://us04web.zoom.us/j/77390581512?pwd=ACcYnHug7H7bGGUvTnOuaPC9ZYcZOw.1']
    # Choose a random URL from the list
    chosen_url = random.choice(url_list)

    # Parse the URL
    parsed_url = urllib.parse.urlparse(chosen_url)

    # Extract the path and query components
    path = parsed_url.path
    query = urllib.parse.parse_qs(parsed_url.query)

    # Extract meeting ID from the path
    meeting_id = path.split('/')[-1]

    password = query.get('pwd', [None])[0]
    print(password)
    return int(meeting_id), password

class TestBot():
    def __init__(self):
        pass

    def init(self):
        init_param = zoom.InitParam()

        init_param.strWebDomain = "https://zoom.us"
        init_param.strSupportUrl = "https://zoom.us"
        init_param.enableGenerateDump = True
        init_param.emLanguageID = zoom.SDK_LANGUAGE_ID.LANGUAGE_English
        init_param.enableLogByDefault = True

        init_sdk_result = zoom.InitSDK(init_param)
        
        if init_sdk_result != zoom.SDKERR_SUCCESS:     
            raise Exception('InitSDK failed')

        self.meeting_service_event = zoom.MeetingServiceEventCallbacks(onMeetingStatusChangedCallback=self.meeting_status_changed)
        self.meeting_service = zoom.CreateMeetingService()  
        self.meeting_service.SetEvent(self.meeting_service_event)

     
        self.auth_event = zoom.AuthServiceEventCallbacks(onAuthenticationReturnCallback=self.auth_return)
        self.auth_service = zoom.CreateAuthService()
        self.auth_service.SetEvent(self.auth_event)
    
        # Use the auth service
        auth_context = zoom.AuthContext()
        ZOOM_APP_CLIENT_ID="fhoqlBo1QFyEk8kU7Wwyhg"
        ZOOM_APP_CLIENT_SECRET="zs8olDayzNYGmgMBHcDGBZnYQvEuDZMP"
        auth_context.jwt_token = generate_jwt(ZOOM_APP_CLIENT_ID, ZOOM_APP_CLIENT_SECRET)
        self.jwt_token=auth_context.jwt_token
        result = self.auth_service.SDKAuth(auth_context)
        if result != zoom.SDKError.SDKERR_SUCCESS:
            raise Exception('SDKAuth failed!')
        print('&&&&&&&&&&&&&&&&&&&&&&&')

        
    def meeting_status_changed(self, status, iResult):
        print(status,'$$$$$$$$$$$$$$$$$$$$$$$$$$', iResult)
        print('------------------------------')
        if status == zoom.MEETING_STATUS_INMEETING:
            print("joined meeting")

            my_user_name = self.meeting_service.GetMeetingParticipantsController().GetMySelfUser().GetUserName()
            print(my_user_name)
            if my_user_name != "TestJoinBot":
                raise Exception("Failed to get username")

            with open("/tmp/test_passed", 'w') as f:
                print('test_passed')
                f.write('test_passed')
            GLib.timeout_add_seconds(60, self.exit_process)
            # Zoom API endpoint to fetch meeting recordings

            


    def exit_process(self):
        print('I am exitting')
        if main_loop:
            main_loop.quit()
        print('I am exitting')
        return False  # To stop the timeout from repeating

    def auth_return(self, result):
        print(result)
        if result == zoom.AUTHRET_SUCCESS:
            print("Auth completed successfully.")

            meeting_number, password = get_random_meeting()

            display_name = "TestJoinBot"

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

            self.meeting_service.Join(join_param)
            return

        raise Exception("Failed to authorize. result = ", result)

    def leave(self):
        if self.meeting_service is None:
            return
        
        if self.meeting_service.GetMeetingStatus() == zoom.MEETING_STATUS_IDLE:
            return
        
        self.meeting_service.Leave(zoom.LEAVE_MEETING)

    def cleanup(self):
        if self.meeting_service:
            print("DestroyMeetingService")
            zoom.DestroyMeetingService(self.meeting_service)
            print("EndDestroyMeetingService")

        if self.auth_service:
            print("DestroyAuthService")
            zoom.DestroyAuthService(self.auth_service)
            print("EndDestroyAuthService")

        print("CleanUPSDK")
        zoom.CleanUPSDK()
        print("EndCleanUPSDK")

def on_signal(signum, frame):
    print(f"Received signal {signum}")
    sys.exit(0)

def on_exit():
    print("Exiting...")
    testBot.leave()
    print("cleaning...")
    testBot.cleanup()

def on_timeout():
    return True  # Returning True keeps the timeout active

def run():
    global testBot, main_loop
    testBot = TestBot()
    testBot.init()   
    
    # Create a GLib main loop
    main_loop = GLib.MainLoop()

    # Add a timeout function that will be called every 100ms
    GLib.timeout_add(100, on_timeout)

    # Run the main loop
    try:
        main_loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    finally:
        main_loop.quit()

def main():
    load_dotenv()

    # Set up signal handlers
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Set up exit handler
    import atexit
    atexit.register(on_exit)

    # Run the Meeting Bot
    run()

import jwt  # PyJWT
import requests
import time


import requests
from requests.auth import HTTPBasicAuth

# Replace these with values from your Zoom App Marketplace


# import requests
# from requests.auth import HTTPBasicAuth


# client_id = '3qzforgxTTqleE3wQxr3Uw'
# client_secret = 'YEGaJystFOA2BTb7YqzUVNtPMF2GZULq'
# account_id = 'roRSAc3jTsSKeBOIjGlCug'

# url = "https://zoom.us/oauth/token"
# params = {
#     "grant_type": "account_credentials",
#     "account_id": account_id
# }

# response = requests.post(
#     url,
#     params=params,
#     auth=HTTPBasicAuth(client_id, client_secret)
# )

# # Print full error message for diagnosis
# if response.status_code != 200:
#     print("Error:", response.status_code)
#     print("Response:", response.text)
#     response.raise_for_status()

# access_token = response.json()['access_token']
# print("Access Token:", access_token)



# import requests


# user_id = "me"  # or Zoom user email / ID
# url = f"https://api.zoom.us/v2/users/{user_id}/meetings"

# headers = {
#     "Authorization": f"Bearer {access_token}",
#     "Content-Type": "application/json"
# }

# payload = {
#     "topic": "My Test Meeting",
#     "type": 1,  # 1 = Instant, 2 = Scheduled, 3 = Recurring
#     "settings": {
#         "host_video": True,
#         "participant_video": True
#     }
# }

# response = requests.post(url, json=payload, headers=headers)
# print(response.json())




if __name__ == "__main__":
    #main()
            url = f"https://api.zoom.us/v2/meetings/77390581512/recordings"
            headers = {"Authorization": f"Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBLZXkiOiJmaG9xbEJvMVFGeUVrOGtVN1d3eWhnIiwiaWF0IjoxNzQ3MTMzNDY3LCJleHAiOjE3NDczMDYyNjcsInRva2VuRXhwIjoxNzQ3MTQwNjY3fQ.l5b88-SwqzLfyzJr53kZjHcx_yPVlifaCQrnfsKT7fs"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                recording_data = response.json()
                recording_files = recording_data.get("recording_files", [])
                for file in recording_files:
                    if file['file_type'] == 'MP4':  # Video recording
                        print( file['download_url'])
            else:
                print(f"Error fetching recording details: {response.status_code}")

            print('*************************')