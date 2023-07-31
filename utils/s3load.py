# Create an S3 client with credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id='AKIA2JHUK4EGBAMYAYFY',
    aws_secret_access_key='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'
)


# Specify the bucket name and prefix (directory path)
bucket_name = 'anyoneai-datasets'
prefix = 'nasdaq_annual_reports/'

# Specify the local directory path to save the files
local_directory = 'dataset3'

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

# Function to download files with pagination
def download_files_with_pagination(bucket_name, prefix, local_directory):
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in page_iterator:
        for obj in page['Contents']:
            key = obj['Key']

            # Skip directories
            if key.endswith('/'):
                continue

            local_file_path = os.path.join(local_directory, os.path.basename(key))

            # Download the file from S3 to the local path
            s3_client.download_file(bucket_name, key, local_file_path)

            print(f'Downloaded: {key} -> {local_file_path}')

# Call the function to download all files
download_files_with_pagination(bucket_name, prefix, local_directory)
