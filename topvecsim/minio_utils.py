import os
import boto3


class Minio:
    def __init__(self, bucket: str, max_retries: int = 5):
        """Initialize the MinIO connection to a specific bucket. Technically this can be S3 too, there's
        nothing different functionally.
        """

        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url="https://castle-minio.community.saturnenterprise.io",
            config=boto3.session.Config(signature_version="s3v4"),
        )
        self.s3_bucket = self.s3_resource.Bucket(bucket)

        assert self.s3_bucket.creation_date, ValueError(
            "This MinIO bucket does not exist."
        )

        self.max_retries = max_retries

    def upload_from_path_to_key(self, path: str, key: str):
        """Upload file to MinIO from local path."""

        current_try = 0
        while current_try <= self.max_retries:
            try:
                self.s3_bucket.upload_file(path, key)

                return key

            except Exception as e:
                current_try += 1

        raise Exception(
            f"Failed to upload file: {path} to {key} after {self.max_retries} tries."
        )

    def download_to_path_from_key(self, key: str, path: str):
        """Download file from a Minio key to a local path."""

        current_try = 0
        while current_try <= self.max_retries:
            try:
                self.s3_bucket.download_file(key, path)

                return path

            except Exception as e:
                current_try += 1

        raise Exception(
            f"Failed to download file: {path} from {key} after {self.max_retries} tries."
        )


# Just checking whether the env variables are available to allow a connection to MinIO.
if os.getenv("AWS_ACCESS_KEY_ID"):
    try:
        minio_client = Minio("topvecsim")
    except Exception as e:
        minio_client = None
else:
    minio_client: Minio = None
