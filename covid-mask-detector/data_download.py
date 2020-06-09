from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path


class DataDownloader:
    def __init__(self, path_to_save, file_id):
        self.path_to_save = path_to_save
        self.file_id = file_id

    def download(self):
        dataset_path = Path(self.path_to_save)
        google_drive_file_id = self.file_id
        gdd.download_file_from_google_drive(file_id=google_drive_file_id,
                                            dest_path=str(dataset_path),
                                            unzip=True)
        dataset_path.unlink()
