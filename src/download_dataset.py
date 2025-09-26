import os
import kaggle
from pathlib import Path
import logging


def download_kaggle_dataset(competition_name, download_path):
    """
    Download dataset from Kaggle competition
    """
    Path(download_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading {competition_name} dataset to {download_path}")

    try:
        kaggle.api.competition_download_files(
            competition=competition_name, path=download_path, quiet=False
        )

        # Extract the zip file
        import zipfile

        zip_path = os.path.join(download_path, f"{competition_name}.zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_path)

        # Remove the zip file
        os.remove(zip_path)

        logging.info("Download and extraction completed successfully!")

        # List downloaded files
        files = os.listdir(download_path)
        logging.info(f"Downloaded files: {files}")

    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    import sys

    competition = sys.argv[1] if len(sys.argv) > 1 else "home-credit-default-risk"
    download_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw"

    download_kaggle_dataset(competition, download_path)
