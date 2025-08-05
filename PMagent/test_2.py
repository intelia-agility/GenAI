import google.cloud.aiplatform as aiplatform
from typing import Union, Sequence
import vertexai


def create_batch_prediction_job_dedicated_resources_sample(project: str,location: str,model_resource_name: str,job_display_name: str,gcs_source: Union[str, Sequence[str]], gcs_destination: str,instances_format: str = "jsonl",machine_type: str = "n1-standard-2", accelerator_count: int = 1, accelerator_type: str = "NVIDIA_TESLA_K80", starting_replica_count: int = 1, max_replica_count: int = 1, sync: bool = True,):
    aiplatform.init(project=project, location=location)
    print(aiplatform.Model.list())
    print('*******')
    my_model = aiplatform.Model(model_resource_name)

    batch_prediction_job = my_model.batch_predict(
        job_display_name=job_display_name,
        instances_format=instances_format,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination,
        #machine_type=machine_type,
        #accelerator_count=accelerator_count,
        #accelerator_type=accelerator_type,
        starting_replica_count=starting_replica_count,
        max_replica_count=max_replica_count,
        sync=sync,
    )

    return batch_prediction_job


GOOGLE_CLOUD_PROJECT='nine-search-gen-project'
GOOGLE_CLOUD_REGION="us-central1"
output_uri ="gs://nine_dry_run_showcase_assets/test/image_batch_prediction_fldr_out"
input_uri="gs://nine_dry_run_showcase_assets/test/image_batch_request_fldr_image_batch_request_fldr20241030T023903797355Z_image_request_20241030024024_0.json"
 
# create_batch_prediction_job_dedicated_resources_sample(
#     project=GOOGLE_CLOUD_PROJECT,
#     location=GOOGLE_CLOUD_REGION,
#     model_resource_name="gemini-1.5-pro-002",
#     job_display_name=f"batch_test",
#     gcs_source=input_uri,
#     gcs_destination=output_uri,
#     accelerator_type=None,
#     accelerator_count=None,
#     sync=False,
# )

from vertexai.preview.language_models import TextEmbeddingModel
vertexai.init(project='nine-search-gen-project', location="us-central1")

textembedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
batch_prediction_job = textembedding_model.batch_predict(
  dataset=[input_uri],
  destination_uri_prefix=output_uri,
)
print(batch_prediction_job.display_name)
print(batch_prediction_job.resource_name)
print(batch_prediction_job.state)