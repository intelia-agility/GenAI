from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def create_batch_prediction_job_sample(
    project: str,
    display_name: str,
    model_name: str,
    instances_format: str,
    gcs_source_uri: str,
    predictions_format: str,
    gcs_destination_output_uri_prefix: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    model_parameters_dict = {}
    model_parameters = json_format.ParseDict(model_parameters_dict, Value())

    batch_prediction_job = {
        "display_name": display_name,
        # Format: 'projects/{project}/locations/{location}/models/{model_id}'
        "model": model_name,
        "model_parameters": model_parameters,
        "input_config": {
            "instances_format": instances_format,
            "gcs_source": {"uris": [gcs_source_uri]},
        },
        "output_config": {
            "predictions_format": predictions_format,
            "gcs_destination": {"output_uri_prefix": gcs_destination_output_uri_prefix},
        } 
        #,
        # "dedicated_resources": {
        #     "machine_spec": {
        #         "machine_type": "n1-standard-2",
        #         "accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
        #         "accelerator_count": 1,
        #     },
        #     "starting_replica_count": 1,
        #     "max_replica_count": 1,
        #},
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.create_batch_prediction_job(
        parent=parent, batch_prediction_job=batch_prediction_job
    )
    print("response:", response)

model_name="publishers/google/models/gemini-1.5-flash-002"
model_id='gemini-1.5-pro-002'
GOOGLE_CLOUD_PROJECT='nine-search-gen-project'
GOOGLE_CLOUD_REGION="us-central1"
output_uri ="gs://nine_dry_run_showcase_assets/test/image_batch_prediction_fldr_out"
input_uri="gs://nine_dry_run_showcase_assets/test/image_batch_request_fldr_image_batch_request_fldr20241030T023903797355Z_image_request_20241030024024_0.json"
#model_name=  'projects/nine-search-gen-project/locations/us-central1/models/gemini-1.5-pro-002'

 
x=create_batch_prediction_job_sample(
    project=GOOGLE_CLOUD_PROJECT,
    display_name='test',
    model_name=model_name,
    instances_format='jsonl',
    gcs_source_uri=input_uri,
    predictions_format='jsonl',
    gcs_destination_output_uri_prefix=output_uri,
    location = "us-central1",
    api_endpoint ="us-central1-aiplatform.googleapis.com" 
)
print(x)

# api_endpoint= "us-central1-aiplatform.googleapis.com"
# # The AI Platform services require regional API endpoints.
# client_options = {"api_endpoint": api_endpoint}
# # Initialize client that will be used to create and send requests.
# # This client only needs to be created once, and can be reused for multiple requests.
# client = aiplatform.gapic.JobServiceClient(client_options=client_options)
# #print(client.list_batch_prediction_jobs())

# aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
# model = aiplatform.Model(model_name='gemini-1.5-pro-002', project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
 