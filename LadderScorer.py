from collections import deque
from statistics import mean
from sklearn.cluster import KMeans

import os
import yaml
import cv2
import os
import numpy as np
import pandas as pd
import subprocess as sp
import matplotlib.pyplot as plt

class LadderAnalysis:
    def __init__(self, config_path=None, ID=None, full_vid_filename=None, snapshot=None, FPS=30, cropping=False, videofileType='.mp4'):
        self.config_path = config_path
        self.ID = ID
        self.filename = full_vid_filename
        self.snapshot = snapshot
        self.FPS = FPS
        self.cropping = cropping
        self.videoType = videofileType
        self.features = ["Nose_likelihood", "TailBase_likelihood", "BackLeft_likelihood", "FrontLeft_likelihood", "FrontRight_likelihood", "BackRight_likelihood"]
        self.limbs = ['FrontLeft', 'BackLeft', 'FrontRight', 'BackRight']

    def video_shape(self):
        video_file = self.filename
        cap = cv2.VideoCapture(video_file)
        h, w = cap.get(4), cap.get(3)
        return video_file, h, w

    def getBasedir(self):
        base_dir = self.filename.split(self.filename.split("/")[-1])[0]
        return base_dir

    def get_first_frame(self, show=False):
        filename = self.filename
        cap = cv2.VideoCapture(filename)
        firstFrame = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                firstFrame.append(img_gray)
                break
        cap.release()
        if show:

            return firstFrame
        else:
            return firstFrame

    def cropCheck(self, cropCheck=None):
        firstFrame = self.get_first_frame()
        if cropCheck:
            x1, x2, y1, y2 = cropCheck
            plt.figure(figsize = (40, 10))
            plt.imshow(firstFrame[0][y1: y2, x1: x2])
        else:
            h,w = self.video_shape()[1:]
            plt.imshow(firstFrame[0][0: int(h), 0: int(w)])

    def updateConfigforCropping(self, croppingParams=None):
        with open(self.config_path, 'r') as f:
            try:
                editedYAML = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        if croppingParams:
            x1, x2, y1, y2 = croppingParams
            editedYAML['cropping'] = True
            editedYAML['x1'] = x1
            editedYAML['x2'] = x2
            editedYAML['y1'] = y1
            editedYAML['y2'] = y2
            with open(config_path, "w") as f:
                yaml.dump(editedYAML, f)
            return editedYAML
        else:
            return editedYAML

    def analyseVideo(self):
        try:
            import deeplabcut
            os.environ["DLClight"]="True"
            os.environ["Colab"]="True"
            deeplabcut.analyze_videos(self.config_path, [self.filename], videotype=self.videoType, save_as_csv=True, shuffle=1)
        except ImportError:
            sp.call(['pip', 'install', 'deeplabcut'])
            print("DeepLabCut (DLC) has been installed -- restart runtime to correctly load dependencies for DLC-Pose-Estimation")

    def checkLabels(self):
        try:
            import deeplabcut
            os.environ["DLClight"]="True"
            os.environ["Colab"]="True"
            deeplabcut.create_labeled_video(self.config_path, [self.filename])
        except ImportError:
            sp.call(['pip', 'install', 'deeplabcut'])
            print("DeepLabCut (DLC) has been installed -- restart runtime to correctly load dependencies for DLC-Pose-Estimation")

    def get_csv_filename(self):
        directory = os.fsencode(self.getBasedir())
        for file in os.listdir(directory):
            filename_ = os.fsdecode(file)
            if filename_.endswith(".csv"):
                if animal.filename.split(".mp4")[0].split(r"/")[-1] in filename_:
                    return "{}{}".format(self.getBasedir(), filename_)
        return "No CSV for this video; try running analyseVideo()."

    def clean_data(self):
        file = self.get_csv_filename()
        data = pd.read_csv(file)
        data.columns = data.iloc[0]
        new = data[2:]
        new.set_index("bodyparts", inplace=True)
        vals = list(range(1, 21, 3))
        new_cols = []
        for i in enumerate(new.columns):
            x, y = i
            adjusted = x + 1
            if adjusted % 3 == 0:
                new_cols.append("{}_likelihood".format(y))
            elif adjusted not in vals:
                new_cols.append("{}_y".format(y))
            else:
                new_cols.append("{}_x".format(y))
        new.columns = new_cols
        return new

    def get_animal_presence(self): #performs k-means clustering on the likelihood scores of all of the features and puts them in two clusters
        new = self.clean_data()
        series_list = []
        for i in self.features:
            ser = new[i].astype("float").rolling(window=100, min_periods=1).mean()
            series_list.append(ser)
        animal = pd.concat(series_list, axis=1)
        clusters = 2
        kmeans = KMeans(n_clusters = clusters)
        kmeans.fit(animal)
        final = []
        if kmeans.labels_[0] == 1: #makes sure the labels of the clusters are the same each time, where 1 == present
            for i in kmeans.labels_:
                if i == 1:
                    final.append(0)
                else:
                    final.append(1)
            return final
        else:
            return kmeans.labels_

    def bodylength(self):
        cleaned = self.clean_data()
        cleaned = cleaned[['TailBase_likelihood','Nose_likelihood', 'TailBase_x','TailBase_y' , 'Nose_x','Nose_y']].copy()
        bodyLength = cleaned[(cleaned['TailBase_likelihood'].astype("float") > 0.99) & (cleaned['Nose_likelihood'].astype("float") > 0.99)]
        bodyLength["BodyLengthSquared"] = np.square(bodyLength["TailBase_x"].astype("float") - bodyLength["Nose_x"].astype("float")) + np.square(bodyLength["TailBase_y"].astype("float") - bodyLength["Nose_y"].astype("float"))
        bodyLength["BodyLength"] = np.sqrt(bodyLength["BodyLengthSquared"])
        avg_bodyLength = np.mean(bodyLength["BodyLength"])
        return avg_bodyLength, bodyLength

    def plot_presence(self):
        nose =  self.get_animal_presence()
        plt.plot(nose)

    def slices_and_runs(self):
        nose =  self.get_animal_presence()
        changes = deque(maxlen=2)
        latency = 0
        run_lengths = []
        slices = []
        val = 0
        first = True
        up = False
        for i in nose.Smoothed:
            val += 1
            changes.append(i)
            if i == 1:
                latency += 1
            if list(changes) == [1, 0]:
                if up:
                    slices.append(val)
                    run_lengths.append(int(latency / 50))
                    latency = 0
                up = False
            if list(changes) == [0, 1]:
                slices.append(val)
                up = True

        if list(changes) == [1, 1]:
            slices.append(val)
            run_lengths.append(int(latency / FPS))
        return slices, run_lengths

    def best_fit_slope_and_intercept(self, xs, ys): #returns the gradient and y intercept of the line of best fit through a set of coordinates
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
           ((mean(xs) * mean(xs)) - mean(xs * xs)))

        b = mean(ys) - m * mean(xs)
        return m, b

    def get_nose_line_equation(self): #get the gradient and y intercept of the mouse's nose whereabouts
        new = self.clean_data()
        slices = self.slices_and_runs()[0]
        fit_x = []
        fit_y = []
        run_v = 0
        for run in range(int(len(slices) / 2)):
            for i in range(slices[run_v], slices[run_v + 1]):
                if new.iloc[i].astype("float").Nose_likelihood > 0.9:
                    fit_x.append(new.iloc[i].astype('float').Nose_x)
                    fit_y.append(new.iloc[i].astype('float').Nose_y)
            run_v += 2
        m, b = self.best_fit_slope_and_intercept(np.array(fit_x), np.array(fit_y))
        return m, b

    def clean_slices(self): #get rid of any runs that register as under 4 seconds in length
        slices = self.slices_and_runs()[0]
        unit = 0
        for i in range(int(len(slices)/2)):
            length = int((slices[unit+1]-slices[unit])/50)
            if length < 4:
                slices[unit] = "n"
                slices[unit+1] = "n"
            unit += 2
        new_slices = []
        for i in slices:
            if i != 'n':
                new_slices.append(i)
        return new_slices

    def distanceFromRungs(self):
        pass

    def distanceFromMidpoint(self):
        pass