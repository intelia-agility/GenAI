from google.cloud import aiplatform


import time
import vertexai

from vertexai.preview.batch_prediction import BatchPredictionJob
output_uri = 'gs://gen_ai_trigger_us_west/image_batch_prediction_fldr_out'

# TODO(developer): Update and un-comment below line
# PROJECT_ID = "your-project-id"

location='us-west1'
project='genai-trigger-test'

# Initialize vertexai
vertexai.init(project=project , location=location)

#input_uri = "gs://cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl"
input_uri='gs://gen_ai_trigger_us_west/test_image.json'
# Submit a batch prediction job with Gemini model
batch_prediction_job = BatchPredictionJob.submit(
    source_model="gemini-1.5-flash-002",
    input_dataset=input_uri,
    output_uri_prefix=output_uri,
)

# Check job status
print(f"Job resource name: {batch_prediction_job.resource_name}")
print(f"Model resource name with the job: {batch_prediction_job.model_name}")
print(f"Job state: {batch_prediction_job.state.name}")

# Refresh the job until complete
while not batch_prediction_job.has_ended:
    time.sleep(5)
    batch_prediction_job.refresh()

# Check if the job succeeds
if batch_prediction_job.has_succeeded:
    print("Job succeeded!")
else:
    print(f"Job failed: {batch_prediction_job.error}")

# Check the location of the output
print(f"Job output location: {batch_prediction_job.output_location}")

# Example response:
# 