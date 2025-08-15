import time
import vertexai

from vertexai.preview.batch_prediction import BatchPredictionJob

# TODO(developer): Update and un-comment below lines
# input_uri ="gs://[BUCKET]/[OUTPUT].jsonl" # Example
# 
vertexai.init(project='nine-search-gen-project', location="us-central1")

output_uri ="gs://nine_dry_run_showcase_assets/test/image_batch_prediction_fldr_out"
input_uri="gs://nine_dry_run_showcase_assets/test/image_batch_request_fldr_image_batch_request_fldr20241030T023903797355Z_image_request_20241030024024_0.json"
 
# Submit a batch prediction job with Gemini model
batch_prediction_job = BatchPredictionJob.submit(
    source_model="gemini-1.5-pro-002",
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
#  Job output location: gs://your-bucket/gen-ai-batch-prediction/prediction-model-year-month-day-hour:minute:second.12345

# https://storage.googleapis.com/cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl
 