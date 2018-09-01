import numpy as np
import os
import shutil
from video_to_frames import extractImages
import glob
import cv2
import matplotlib.pyplot as plt
import time

class DataProvider:

    def __init__(self, video_name, img_sz):

        # self.config = config
        self.feed_path = "Data"

        # load data from video
        makedir(self.feed_path)
        feed_size = extractImages(video_name, self.feed_path)
        #feed_size = 8900

        self.train_size = int(0.8 * feed_size)
        self.test_size = feed_size - self.train_size
        self.train = []
        self.test = []

        print('Start loading data ...')
        files = glob.glob(self.feed_path + "/*.jpg")
        count = 1
        for img_str in files:
            I = cv2.imread(img_str)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            I = cv2.resize(I, (img_sz, img_sz))
            I = I / 255.
            I = np.atleast_3d(I)
            if count <= self.train_size:
                self.train.append(I)
            else:
                self.test.append(I)
            count += 1

        self.train = np.array(self.train)
        self.test = np.array(self.test)
        print('Finished uploading data, Train data shape:', self.train.shape, '; Test data shape:', self.test.shape)

    def next_batch(self, batch_size, data_type):
        batch_img = None
        if data_type == 'train':
            idx = np.random.choice(self.train_size, batch_size)
            batch_img = self.train[idx, ...]
        elif data_type == 'test':
            idx = np.random.choice(self.test_size, batch_size)
            batch_img = self.test[idx, ...]
        return batch_img


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
    # cd into the specified directory
    # os.chdir(folder_name)