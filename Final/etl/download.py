# import boto3
# from getpass import getpass
# import os 

import boto3
import os
from config import *
# Define AWS credentials
aws_access_key_id = 'YOUR_ACCESS_KEY_ID'
aws_secret_access_key = 'YOUR_SECRET_ACCESS_KEY'

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Create the paginator for the list_objects_v2 API
paginator = s3_client.get_paginator('list_objects_v2')

# Specify the bucket name and prefix (directory path)
bucket_name = 'your-bucket-name'
prefix = 'your-prefix/'

# List objects within the bucket using the paginator, limiting to 999 objects
response_iterator = paginator.paginate(
    Bucket=bucket_name,
    Prefix=prefix,
    PaginationConfig={'MaxItems': 1000}
)

# Specify the local directory path to save the files
local_directory = Local_directory

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

# Download each file to the local directory
for response in response_iterator:
    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            
            # Skip directories
            if key.endswith('/'):
                continue
            
            local_file_path = os.path.join(local_directory, os.path.basename(key))
            
            # Download the file from S3 to the local path
            s3_client.download_file(bucket_name, key, local_file_path)
            
            print(f'Downloaded: {key} -> {local_file_path}')