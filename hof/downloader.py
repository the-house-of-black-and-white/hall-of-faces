import logging
import os
import tarfile
import zipfile

import requests

log = logging.getLogger(__name__)

CHUNK_SIZE = 32768


class Downloader:

    def __init__(self, target_dir):
        self.target_dir = target_dir

    def download_file_from_web_server(self, url, destination):
        log.info('Downloading {} into {}'.format(url, destination))
        local_filename = url.split('/')[-1]
        target_file = os.path.join(destination, local_filename)
        if not os.path.exists(target_file):
            response = requests.get(url, stream=True)
            self.save_response_content(response, target_file)
        log.info('Finished download')
        return local_filename

    def download_file_from_google_drive(self, id, destination):
        log.info('Downloading from Google Drive id={} into {}'.format(id, destination))
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)

        log.info('Finished download')

    @staticmethod
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    @staticmethod
    def extract_zip_file(zip_file_name, destination):
        log.info('Extracting {} into {}'.format(zip_file_name, destination))
        zip_ref = zipfile.ZipFile(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()

    @staticmethod
    def extract_tar_file(zip_file_name, destination):
        log.info('Extracting {} into {}'.format(zip_file_name, destination))
        zip_ref = tarfile.TarFile.open(zip_file_name, 'r')
        zip_ref.extractall(destination)
        zip_ref.close()
