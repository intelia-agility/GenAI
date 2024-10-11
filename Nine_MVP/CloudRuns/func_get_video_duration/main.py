import functions_framework
import gcsfs
from pymediainfo import MediaInfo

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the input data model
class VideoRequest(BaseModel):
    url: str  # Input: video URL

# Define the output data model
class VideoResponse(BaseModel):
    duration: float  # Output: Duration in seconds

@app.post("/get-video-duration", response_model=VideoResponse)
async def get_video_duration(request: VideoRequest):
    """
    get video duration of a given gcs url
    
    """
    
    fs = gcsfs.GCSFileSystem()
    # Open the file stream using gcsfs
    with fs.open(request.url, 'rb') as video_file:
              # Use pymediainfo to extract metadata directly from the stream
              media_info = MediaInfo.parse(video_file)
              for track in media_info.tracks:
                  if track.track_type == 'Video':
                      duration= track.duration / 1000  # Convert ms to seconds
                      print(duration)
                      break
 

    return VideoResponse(duration=duration)

