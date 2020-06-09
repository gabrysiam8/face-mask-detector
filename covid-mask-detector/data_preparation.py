from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2


# download dataset from link provided by
# https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
from data_download import DataDownloader

downloader = DataDownloader(path_to_save='covid-mask-detector/data/mask.zip', file_id='1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp')
downloader.download()

maskDF = pd.DataFrame()


def add_photos_to_category(desc, mask_id, maskPath):
    NoneType = type(None)
    global subject, imgPath, maskDF
    for subject in tqdm(list(maskPath.iterdir()), desc=desc):
        for imgPath in subject.iterdir():
            img = cv2.imread(str(imgPath))
            if type(img) != NoneType:
                maskDF = maskDF.append({'image': str(imgPath), 'mask': mask_id}, ignore_index=True)


datasetPath = Path('covid-mask-detector/data/self-built-masked-face-recognition-dataset')
add_photos_to_category('mask photos', 1, datasetPath/'AFDB_masked_face_dataset')
add_photos_to_category('non mask photos', 0, datasetPath/'AFDB_face_dataset')

dfName = 'covid-mask-detector/data/mask_df.pickle'
print(f'saving Dataframe to: {dfName}')
maskDF.to_pickle(dfName, "infer", 3)
