import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from .logger import get_logger

class S3Service:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.region = os.getenv('AWS_REGION', 'ap-south-1')
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'call-iq')

        # Configure boto3 client with specific settings
        config = Config(
            region_name=self.region,
            signature_version='s3v4',  # Use AWS4-HMAC-SHA256
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            }
        )
        self.s3 = boto3.client(
            's3',
            config=config,
            region_name=self.region
        )

    # -------------------------
    # Upload a local file to S3
    # -------------------------
    def upload_file(self, bucket: str, key: str, file_path: str) -> bool:
        try:
            self.s3.upload_file(file_path, bucket, key)
            self.logger.info(f"File uploaded: {file_path} → s3://{bucket}/{key}")
            return True
        except ClientError as e:
            self.logger.error(f"Upload error for {file_path}: {e}", exc_info=True)
            return False

    # -------------------------
    # Download a file from S3
    # -------------------------
    def download_file(self, bucket: str, key: str, download_path: str) -> bool:
        try:
            self.s3.download_file(bucket, key, download_path)
            self.logger.info(f"File downloaded: s3://{bucket}/{key} → {download_path}")
            return True
        except ClientError as e:
            self.logger.error(f"Download error for {key}: {e}", exc_info=True)
            return False

    # -------------------------
    # Get metadata of an S3 object
    # -------------------------
    def get_metadata(self, bucket: str, key: str):
        try:
            response = self.s3.head_object(Bucket=bucket, Key=key)
            self.logger.info(f"Metadata retrieved for s3://{bucket}/{key}")
            return response.get("Metadata", {}), response
        except ClientError as e:
            self.logger.error(f"Metadata error for {key}: {e}", exc_info=True)
            return {}, {}

    # -------------------------
    # Stream upload
    # -------------------------
    def upload_stream(self, bucket: str, key: str, file_bytes: bytes) -> bool:
        try:
            self.s3.put_object(Bucket=bucket, Key=key, Body=file_bytes)
            self.logger.info(f"Stream uploaded to s3://{bucket}/{key}")
            return True
        except ClientError as e:
            self.logger.error(f"Stream upload error for {key}: {e}", exc_info=True)
            return False

    # -------------------------
    # Stream download
    # -------------------------
    def download_stream(self, bucket: str, key: str) -> bytes:
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            self.logger.info(f"Stream downloaded from s3://{bucket}/{key}")
            return response["Body"].read()
        except ClientError as e:
            self.logger.error(f"Stream download error for {key}: {e}", exc_info=True)
            return b""
