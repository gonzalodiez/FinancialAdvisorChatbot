import boto3
from getpass import getpass
import os 



# Create an S3 client with credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id='AKIA2JHUK4EGBAMYAYFY',
    aws_secret_access_key='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'
)

# Specify the bucket name and prefix (directory path)
bucket_name = 'anyoneai-datasets'
prefix = 'nasdaq_annual_reports/'


# List objects within the bucket
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)


# Specify the local directory path to save the files
local_directory = 'dataset'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

# Download each file to the local directory
for obj in response['Contents']:
    key = obj['Key']
    
    # Skip directories
    if key.endswith('/'):
        continue
    
    local_file_path = os.path.join(local_directory, os.path.basename(key))

    # Download the file from S3 to the local path
    s3_client.download_file(bucket_name, key, local_file_path)

    print(f'Downloaded: {key} -> {local_file_path}')