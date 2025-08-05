from google.cloud import documentai_v1 as documentai


import re
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai  # type: ignore
from google.cloud import storage

def extract_text_from_pdf(project_id, location, processor_id, gcs_uri):
    # client = documentai.DocumentUnderstandingServiceClient()
    # name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    # document = {"gcs_source": {"uri": gcs_uri}, "mime_type": "application/pdf"}
    # request = {"name": name, "raw_document": None, "document": document}
    
    # result = client.process_document(request=request)
    # return result.document.text

    from google.cloud import documentai_v1
    from google.cloud.documentai_v1 import types

    client = documentai_v1.DocumentProcessorServiceClient()

    # The full resource name of the processor
    name = f"projects/{project_id}/locations/us/processors/{processor_id}"
    gcs_output_uri="gs://pmtask_tracker/"
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri, field_mask=None
    )
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)


    # Configure the GCS input
    gcs_document = documentai_v1.GcsDocument(
        gcs_uri=gcs_uri,
        mime_type="application/pdf"
    )
    input_config = documentai_v1.BatchDocumentsInputConfig(
        gcs_documents=documentai_v1.GcsDocuments(documents=[gcs_document])
    )

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    ) 
   

    # NOTE: This only works with synchronous processors like OCR or form parser
    op = client.batch_process_documents(request=request)

    # Print the raw extracted text
    #document = result.d
    #print("Extracted Text:\n")
      # Continually polls the operation until it is complete.
    # This could take some time for larger files
    # Format: projects/{project_id}/locations/{location}/operations/{operation_id}
    try:
        print(f"Waiting for operation {op.operation.name} to complete...")
        op.result(timeout=1000)
    # Catch exception when operation doesn't finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)


    return result


from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import InternalServerError
from google.api_core.exceptions import RetryError
from google.cloud import documentai  # type: ignore
from google.cloud import storage

# TODO(developer): Uncomment these variables before running the sample.
# project_id = "YOUR_PROJECT_ID"
# location = "YOUR_PROCESSOR_LOCATION" # Format is "us" or "eu"
# processor_id = "YOUR_PROCESSOR_ID" # Create processor before running sample
# gcs_output_uri = "YOUR_OUTPUT_URI" # Must end with a trailing slash `/`. Format: gs://bucket/directory/subdirectory/
# processor_version_id = "YOUR_PROCESSOR_VERSION_ID" # Optional. Example: pretrained-ocr-v1.0-2020-09-23

# TODO(developer): You must specify either `gcs_input_uri` and `mime_type` or `gcs_input_prefix`
# gcs_input_uri = "YOUR_INPUT_URI" # Format: gs://bucket/directory/file.pdf
# input_mime_type = "application/pdf"
# gcs_input_prefix = "YOUR_INPUT_URI_PREFIX" # Format: gs://bucket/directory/
# field_mask = "text,entities,pages.pageNumber"  # Optional. The fields to return in the Document object.


def batch_process_documents(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_output_uri: str,
    processor_version_id: Optional[str] = None,
    gcs_input_uri: Optional[str] = None,
    input_mime_type: Optional[str] = None,
    gcs_input_prefix: Optional[str] = None,
    field_mask: Optional[str] = None,
    timeout: int = 10000,
) -> None:
    # You must set the `api_endpoint` if you use a location other than "us".
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if gcs_input_uri:
        # Specify specific GCS URIs to process individual documents
        gcs_document = documentai.GcsDocument(
            gcs_uri=gcs_input_uri, mime_type=input_mime_type
        )
        ##############
        # This line is where the single document is being added
        ##############
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        # Specify a GCS URI Prefix to process an entire directory
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    # Cloud Storage URI for the Output Directory
    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri, field_mask=field_mask
    )

    # Where to write results
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    if processor_version_id:
        # The full resource name of the processor version, e.g.:
        # projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}
        name = client.processor_version_path(
            project_id, 'us', processor_id, processor_version_id
        )
    else:
        # The full resource name of the processor, e.g.:
        # projects/{project_id}/locations/{location}/processors/{processor_id}
        name = client.processor_path(project_id, 'us', processor_id)

    name = f"projects/{project_id}/locations/us/processors/{processor_id}"
    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )
    
    # BatchProcess returns a Long Running Operation (LRO)
    operation = client.batch_process_documents(request)

    # Continually polls the operation until it is complete.
    # This could take some time for larger files
    # Format: projects/{project_id}/locations/{location}/operations/{operation_id}
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)
    # Catch exception when operation doesn't finish before timeout
    except (RetryError, InternalServerError) as e:
        print(e.message)

    # NOTE: Can also use callbacks for asynchronous processing
    #
    # def my_callback(future):
    #   result = future.result()
    #
    # operation.add_done_callback(my_callback)

    # Once the operation is complete,
    # get output document information from operation metadata
    metadata = documentai.BatchProcessMetadata(operation.metadata)

    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    storage_client = storage.Client()

    print("Output files:")
    # One process per Input Document
    for process in list(metadata.individual_process_statuses):
        # output_gcs_destination format: gs://BUCKET/PREFIX/OPERATION_NUMBER/INPUT_FILE_NUMBER/
        # The Cloud Storage API requires the bucket name and URI prefix separately
        matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
        if not matches:
            print(
                "Could not parse output GCS destination:",
                process.output_gcs_destination,
            )
            continue

        output_bucket, output_prefix = matches.groups()

        # Get List of Document Objects from the Output Bucket
        output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)

        # Document AI may output multiple JSON files per source file
        for blob in output_blobs:
            # Document AI should only output JSON files to GCS
            if blob.content_type != "application/json":
                print(
                    f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}"
                )
                continue

            # Download JSON File as bytes object and convert to Document Object
            print(f"Fetching {blob.name}")
            document = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )

            # For a full list of Document object attributes, please reference this page:
            # https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document

            # Read the text recognition output from the processor
            print("The document contains the following text:")
            print(document.text)

def extract_entities_from_text(text):
    
    import json
    import requests
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    from google.auth import default
    from google.auth.transport.requests import Request
    import requests

    credentials, project_id = default()
    credentials.refresh(Request())
    token = credentials.token
 
    # Set API endpoint
    url = "https://healthcare.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/YOUR_REGION/services/nlp:analyzeEntities"

    # Prepare the request
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    data = {
        "documentContent": text,
        "documentType": "CLINICAL_NOTES"
    }

    # Make the API call
    response = requests.post(url, headers=headers, json=data)

    # Print the result
    print(json.dumps(response.json(), indent=2))




import uuid
from datetime import datetime

def map_entities_to_fhir(entities):
    patient_id = str(uuid.uuid4())
    fhir_bundle = []

    fhir_bundle.append({
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{"given": ["John"], "family": "Doe"}],
        "gender": "unknown",
        "birthDate": "1970-01-01"
    })

    for entity in entities:
        if entity.type == "MEDICATION":
            fhir_bundle.append({
                "resourceType": "MedicationRequest",
                "id": str(uuid.uuid4()),
                "subject": {"reference": f"Patient/{patient_id}"},
                "authoredOn": datetime.now().date().isoformat(),
                "medicationCodeableConcept": {
                    "text": entity.mention
                },
                "dosageInstruction": [{"text": entity.mention}]
            })
        elif entity.type == "CONDITION":
            fhir_bundle.append({
                "resourceType": "Condition",
                "id": str(uuid.uuid4()),
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {
                    "text": entity.mention
                }
            })

    return fhir_bundle


import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

def send_fhir_to_store(project_id, location, dataset_id, fhir_store_id, fhir_resources, credentials_path):
    SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
    credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    credentials.refresh(Request())
    access_token = credentials.token

    for resource in fhir_resources:
        resource_type = resource['resourceType']
        url = f"https://healthcare.googleapis.com/v1/projects/{project_id}/locations/{location}/datasets/{dataset_id}/fhirStores/{fhir_store_id}/fhir/{resource_type}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/fhir+json"
        }

        response = requests.post(url, headers=headers, json=resource)
        if response.status_code >= 400:
            print("Error uploading:", response.text)
        else:
            print(f"Uploaded {resource_type} successfully.")


#--------------------
 
# Set your config
project_id = "nine-quality-test"
location = "us-central1"  # or your region
processor_id = "6b8a41eb986f2e5b"
gcs_uri = "gs://pmtask_tracker/Download_Sample-Care-Plan_1.pdf"
DATASET_ID = "myhealthcaredataset"
FHIR_STORE_ID = "myfhir_store"
CREDENTIALS_PATH = "path/to/service-account.json"

# Step 1: Extract text
text = extract_text_from_pdf(project_id, location, processor_id, gcs_uri)

# batch_process_documents(
#     project_id,
#     location,
#     processor_id,
#     "gs://pmtask_tracker/",
#     processor_id,
#     gcs_uri,
#     "application/pdf",
#     ".pdf",
#     None,
#     40000
# ) 

print(text)
# Step 2: Extract entities
entities = extract_entities_from_text(text)

# Step 3: Map to FHIR
fhir_data = map_entities_to_fhir(entities)

# Step 4: Upload to Healthcare API
send_fhir_to_store(PROJECT_ID, LOCATION, DATASET_ID, FHIR_STORE_ID, fhir_data, CREDENTIALS_PATH)
