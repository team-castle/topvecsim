import boto3


class Minio:
    def __init__(self, bucket: str, max_retries: int):
        """Initialize the MinIO connection to a specific bucket. Technically this can be S3 too, there's
        nothing different functionally.
        """

        self.s3_resource = boto3.resource("s3")
        self.s3_bucket = self.s3_resource.Bucket(self.bucket)

        assert self.s3_bucket.creation_date, ValueError("This MinIO bucket does not exist.")

        self.max_retries = max_retries

    def upload_from_path_to_key(self, path: str, key: str):
        """Upload file to MinIO from local path.
        """

        current_try = 0
        while current_try <= self.max_retries:
            try:
                self.s3_bucket.upload_file(path, key)

                return key

            except Exception as e:
                current_try += 1

        raise Exception(f"Failed to upload file: {path} to {key}")
