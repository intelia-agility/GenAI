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
input_uri='gs://gen_ai_trigger_us_west/req.json'
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

# Set parameters
GOOGLE_CLOUD_PROJECT = project
GOOGLE_CLOUD_REGION = location
model_id = 'gemini-1.5-pro-002'  # Your model ID
input_uri = 'gs://genai_trigger_test/test_image.json'#'gs://gen_ai_trigger_us_west/req.json'#

# Initialize the AI Platform
aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

# Construct the full model name
model_name = f'projects/{GOOGLE_CLOUD_PROJECT}/locations/{GOOGLE_CLOUD_REGION}/models/{model_id}'
model_name="publishers/google/models/gemini-1.5-flash-002"
# Create the batch prediction job
try:
    job = aiplatform.BatchPredictionJob.create(
        job_display_name='my_batch_prediction_job',
        model_name=model_name,
        gcs_source=input_uri,
        gcs_destination_prefix=output_uri,
        instances_format='jsonl',
        predictions_format='jsonl',
    )
    print(f"Batch Prediction Job created: {job.name}")
except Exception as e:
    print(f"Error creating Batch Prediction Job: {e}")