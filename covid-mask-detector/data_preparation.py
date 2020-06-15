from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
from data_download import DataDownloader


class DataPreparator:
    def __init__(self):
        self.dataframe = pd.DataFrame(columns=['image', 'mask'])

    @staticmethod
    def download_data(data_path, file_id):
        downloader = DataDownloader(path_to_save=data_path, file_id=file_id)
        downloader.download()

    def add_images_to_category(self, category, mask_flag, data_path):
        img_paths = []
        for img_path in tqdm(list(data_path.glob("*/*")), unit='files', desc=category):
            img = cv2.imread(str(img_path))
            if img is not None:
                img_paths.append(str(img_path))
        temp_dataframe = pd.DataFrame({'image': img_paths, 'mask': [mask_flag]*len(img_paths)})
        self.dataframe = self.dataframe.append(temp_dataframe, ignore_index=True)

    def save_dataframe(self, dataframe_path):
        print(f'Saving dataframe to file {dataframe_path}...')
        self.dataframe.to_pickle(dataframe_path, "infer", 3)


if __name__ == '__main__':
    data_preparator = DataPreparator()

    # download dataset from link provided by
    # https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
    data_preparator.download_data(data_path='covid-mask-detector/data/mask.zip',
                                  file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp')

    dataset_path = Path('covid-mask-detector/data/self-built-masked-face-recognition-dataset')
    data_preparator.add_images_to_category('mask photos', 1, dataset_path/'AFDB_masked_face_dataset')
    data_preparator.add_images_to_category('non mask photos', 0, dataset_path/'AFDB_face_dataset')

    data_preparator.save_dataframe('covid-mask-detector/data/mask_df.pickle')
