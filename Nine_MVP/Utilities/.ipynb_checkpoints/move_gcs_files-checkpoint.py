from google.cloud import storage


def move_MAAT_files( dest_bucket_name: str= None, source_bucket_name: str= None,
                          source_folder: str= None, destination_folder: str= None, mime_types: list[str]= None):
    
    # Initialize a client for interacting with GCS
    storage_client = storage.Client()
    # Get the bucket by name
    bucket = storage_client.get_bucket(source_bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs()  
 

    for blob in blobs:
        # Check if the file is an .mp4 file
        if blob.content_type in ["video/mp4"]:
        
            # Create the new blob name for the destination
            temp=blob.name.split("/")
            name=blob.name.split('/')[len(temp)-1]
            if blob.name.startswith("vlt_video_extract/MAAT") :
                dest_folder= f"vlt_video_extract/MAAT/Full"
                                     
                new_blob_name = f"{dest_folder}/{name}"         

                # Copy the blob to the new location
                bucket.copy_blob(blob, bucket, new_blob_name)
                print(f"Copied {blob.name} to {new_blob_name}")


def move_files( dest_bucket_name: str= None, source_bucket_name: str= None,
                          source_folder: str= None, destination_folder: str= None, mime_types: list[str]= None):
    
    # Initialize a client for interacting with GCS
    storage_client = storage.Client()
    # Get the bucket by name
    bucket = storage_client.get_bucket(source_bucket_name)

    # List all objects in the bucket
    blobs = bucket.list_blobs()  
 

    for blob in blobs:
        # Check if the file is an .mp4 file
        if blob.content_type in ["video/mp4"]:
        
            # Create the new blob name for the destination
            temp=blob.name.split("/")
            name=blob.name.split('/')[len(temp)-1]
            if name.upper().startswith("SYD-NINE") or name.upper().startswith("NNNT"):
                dest_folder= f"{destination_folder}/NINE_NIEWS"
            elif name.upper().startswith("MAAT"):
                dest_folder= f"{destination_folder}/MAAT"
            elif name.upper().startswith("60MI"):
                dest_folder= f"{destination_folder}/SIXTY_MINUTES"
            else:
                 dest_folder= f"{destination_folder}/OTHERS"
                     
            new_blob_name = f"{dest_folder}/{name}"         

            # Copy the blob to the new location
            #bucket.copy_blob(blob, bucket, new_blob_name)
            print(f"Copied {blob.name} to {new_blob_name}")

# Example usage
bucket_name = 'bkt-vlt-d-primary-0cg6-video-vms-extract'
source_folder = ''  # Make sure to include the trailing slash
destination_folder = 'vlt_video_extract'

move_MAAT_files(dest_bucket_name=bucket_name,source_bucket_name=bucket_name, source_folder=source_folder, destination_folder=destination_folder,mime_types=[])

 