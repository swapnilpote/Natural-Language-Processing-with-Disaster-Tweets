import os
import zipfile


def download_data(competition: str = "nlp-getting-started") -> list:
    """To download and unzip from kaggle competition data.

    Args:
        competition (str, optional): Competition name of which data needs to
        be downloaded. Defaults to "nlp-getting-started".

    Returns:
        list: Provides list of all files downloaded.
    """
    # Call kaggle download command to get data from platform.
    os.system(f"kaggle competitions download -c {competition}")

    # Unzip and move into data folder.
    folder = os.path.join(os.getcwd(), "data")
    with zipfile.ZipFile(f"{competition}.zip", "r") as zip_ref:
        if not os.path.isdir("data"):
            os.mkdir(folder)

        zip_ref.extractall(folder)

    # Delete zip file.
    if len(os.listdir(folder)):
        os.remove(f"{competition}.zip")

    return os.listdir(folder)


if __name__ == "__main__":
    download_data()
