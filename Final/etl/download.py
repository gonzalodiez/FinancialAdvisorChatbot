import boto3
import os

def download_from_s3():
    # Read AWS credentials from environment variables
    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']

    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    # Create the paginator for the list_objects_v2 API
    paginator = s3_client.get_paginator('list_objects_v2')

    # Read bucket name and prefix from environment variables
    bucket_name = os.environ['BUCKET_NAME']
    prefix = os.environ['PREFIX']

    # List objects within the bucket using the paginator, limiting to 999 objects
    response_iterator = paginator.paginate(
        Bucket=bucket_name,
        Prefix=prefix,
        PaginationConfig={'MaxItems': 1000}
    )

    # Read local directory path from environment variable
    local_directory = os.environ['LOCAL_DIRECTORY']

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

# This allows the module to be run as a standalone script as well
if __name__ == '__main__':
    download_from_s3()
