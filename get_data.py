from google.cloud import storage
from utils import HyperParameters


def download_quickdraw_ndjson(category, destination_file_name):
    """
    Downloads a Quick, Draw! ndjson raw file for a specific category from the public bucket.

    Args:
        category (str): The category of Quick, Draw! data to download (e.g., 'cat', 'dog').
        destination_file_name (str): The local file path where the downloaded file will be saved.
    """
    bucket_name = "quickdraw_dataset"
    source_blob_name = f"full/raw/{category}.ndjson"

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file to the specified local path
    blob.download_to_filename(destination_file_name)

    print(f"Downloaded Quick, Draw! data for '{category}' to {destination_file_name}.")


def download_quickdraw_npz(category_file_name, destination_file_name):
    """
    Downloads a Quick, Draw! numpy bitmap file for a specific category from the public bucket.

    Args:
        category (str): The category of Quick, Draw! data to download (e.g., 'cat', 'dog').
        destination_file_name (str): The local file path where the downloaded file will be saved.
    """
    bucket_name = "quickdraw_dataset"
    source_blob_name = f"sketchrnn/{category_file_name}.npz"

    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file to the specified local path
    blob.download_to_filename(destination_file_name)

    print(
        f"Downloaded Quick, Draw! data for '{category_file_name}' to {destination_file_name}."
    )


if __name__ == "__main__":
    download_quickdraw_npz(
        HyperParameters.DATA_CATEGORY, f"data/{HyperParameters.DATA_CATEGORY}.npz"
    )
