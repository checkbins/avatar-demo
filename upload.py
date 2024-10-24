
def upload_image_to_azure(image_path, azure_connection_string, container_name):
    from azure.storage.blob import BlobServiceClient
    # Initialize the connection to Azure storage account
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=image_path)

    # Upload the created image to Azure Blob Storage
    with open(image_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
        print(f"File {image_path} uploaded to {container_name} as {blob_client.url}.")


def upload_image_to_gcp(image_path, gcp_credentials, bucket_name):
    from google.cloud import storage
    storage_client = storage.Client.from_service_account_json(gcp_credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(image_path)

    with open(image_path, "rb") as data:
        blob.upload_from_file(data)
        print(f"File {image_path} uploaded to {bucket_name} as {blob.public_url}.")

def upload_image_to_s3(image_path, aws_access_key_id, aws_secret_access_key, bucket_name):
    import boto3
    from botocore.exceptions import NoCredentialsError

    # Create an S3 client
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    try:
        s3_client.upload_file(image_path, bucket_name, image_path)
        print(f"File {image_path} uploaded to {bucket_name}.")
    except FileNotFoundError:
        print(f"The file {image_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available for AWS S3.")