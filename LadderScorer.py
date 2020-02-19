from collections import deque
from statistics import mean
from sklearn.cluster import KMeans, MeanShift

import os
import yaml
import cv2
import os
import numpy as np
import pandas as pd
import subprocess as sp
import matplotlib.pyplot as plt

try:
    os.environ["DLClight"]="True"
    os.environ["Colab"]="True"
    import deeplabcut
except ImportError:
    sp.call(['pip', 'install', 'deeplabcut'])
    print("DeepLabCut (DLC) has been installed -- restarting runtime automatically to correctly load dependencies for DLC-Pose-Estimation, including the correct version of PyYaml")
    os.kill(os.getpid(), 9)

class LadderAnalysis:
    def __init__(self, config_path=None, ID=None, full_vid_filename=None, snapshot=None, FPS=200, cropping=False, videofileType='.mp4'):
        self.config_path = config_path
        self.ID = ID
        self.filename = full_vid_filename
        self.snapshot = snapshot
        self.FPS = FPS
        self.cropping = cropping
        self.videoType = videofileType
        self.limbs = ['FrontLeft', 'BackLeft', 'FrontRight', 'BackRight']

    def getFeatures(self):
        bodypartsList = []
        csv = pd.read_csv(self.get_csv_filename())
        for i in csv.iloc[0]:
            if i != "bodyparts":
                if i not in bodypartsList:
                    bodypartsList.append(i)
        return bodypartsList

    def getLimbNames(self):
        pass

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
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            plt.figure(figsize = (20, 5))
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
        editedYAML['iteration'] = 2
        if croppingParams:
            x1, x2, y1, y2 = croppingParams
            editedYAML['cropping'] = True
            editedYAML['x1'] = x1
            editedYAML['x2'] = x2
            editedYAML['y1'] = y1
            editedYAML['y2'] = y2
            with open(self.config_path, "w") as f:
                yaml.dump(editedYAML, f)
            return editedYAML
        else:
            return editedYAML

    def analyseVideo(self):
        deeplabcut.analyze_videos(self.config_path, [self.filename], videotype=self.videoType, save_as_csv=True, shuffle=1)

    def checkLabels(self):
        deeplabcut.create_labeled_video(self.config_path, [self.filename])

    def get_csv_filename(self):
        directory = os.fsencode(self.getBasedir())
        for file_ in os.listdir(directory):
            filename_ = os.fsdecode(file_)
            if filename_.endswith(".csv"):
                if self.filename.lower().split(".mp4")[0].split(r"/")[-1] in filename_.lower():
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
        for i in self.getFeatures():
            ser = new[f'{i}_likelihood'].astype("float").rolling(window=100, min_periods=1).mean()
            series_list.append(ser)
        animal = pd.concat(series_list, axis=1)
        clusters = 2
        kmeans = KMeans(n_clusters = clusters)
        kmeans.fit(animal)
        final = []
        if kmeans.labels_[0] == 1: #makes sure the labels of the clusters are the same each time, where 1 == mouse is  present
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
        cleaned = cleaned[['TailBase_likelihood','Snout_likelihood', 'TailBase_x','TailBase_y' , 'Snout_x','Snout_y']].copy()
        bodyLength = cleaned[(cleaned['TailBase_likelihood'].astype("float") > 0.99) & (cleaned['Snout_likelihood'].astype("float") > 0.99)]
        bodyLength["BodyLengthSquared"] = np.square(bodyLength["TailBase_x"].astype("float") - bodyLength["Snout_x"].astype("float")) + np.square(bodyLength["TailBase_y"].astype("float") - bodyLength["Snout_y"].astype("float"))
        bodyLength["BodyLength"] = np.sqrt(bodyLength["BodyLengthSquared"])
        avg_bodyLength = np.mean(bodyLength["BodyLength"])
        return avg_bodyLength

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
        for i in nose:
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

    def best_fit_slope_and_intercept(self, xs, ys): #courtesy of Sentdex
        #returns the gradient and y intercept of the line of best fit through a set of coordinates
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
                if new.iloc[i].astype("float").Snout_likelihood > 0.9:
                    fit_x.append(new.iloc[i].astype('float').Snout_x)
                    fit_y.append(new.iloc[i].astype('float').Snout_y)
            run_v += 2
        m, b = self.best_fit_slope_and_intercept(np.array(fit_x), np.array(fit_y))
        return m, b

    def plot_rungs(self, x=None, y=None, plotSlip=True, slipthresh=1):
        with open(self.config_path, 'r') as f:
            try:
                editedYAML = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        firstFrame = self.get_first_frame()
        h, w = self.video_shape()[1:]
        m, c = self.get_nose_line_equation()
        #y = mx + c
        img = firstFrame[0][int(editedYAML['y1']):int(editedYAML['y2']), 0:int(w)]
        cv2.line(img, (0, int(c)), (int(w), int(w*m + c)), (0, 255, 0), 9)
        if plotSlip:
            cv2.line(img, (0, int(c+slipthresh)), (int(w), int(w*m + c+ slipthresh)), (255, 0, 0), 9)
        if x and y:
            plt.figure(figsize = (20, 5))
            plt.imshow(img)
            plt.scatter(x, y, marker="o")
        else:
            plt.imshow(img)

    def clean_slices(self, filter=4): #get rid of any runs that register as under 4 seconds in length
        slices = self.slices_and_runs()[0]
        unit = 0
        for i in range(int(len(slices)/2)):
            length = int((slices[unit+1]-slices[unit])/50)
            if length < filter:
                slices[unit] = "n"
                slices[unit+1] = "n"
            unit += 2
        new_slices = []
        for i in slices:
            if i != 'n':
                new_slices.append(i)
        return new_slices

    def meanShiftClustering(self, X, slipthresh):
        if len(X) > 0:
            ms = MeanShift(bandwidth=slipthresh*4)
            ms.fit(np.array(X))
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            #print(cluster_centers)
            n_clusters_ = len(np.unique(labels))
            #print("Number of estimated clusters:", n_clusters_)
            return n_clusters_, cluster_centers
        else:
            return 0

    def meanShiftClusteringOnly(self):
        new = self.clean_data()
        slices = self.clean_slices(filter=filter)
        run_v = 0
        traversal = 1
        m, c = self.get_nose_line_equation()
        slipthresh_ = self.bodylength()/8
        for run in range(int(len(slices)/2)):
            for limb in limbs:
                bliX = []
                for i in range(slices[run_v], slices[run_v + 1]): #iterate through each run and return any coordinates that are below the rung lines with a pcutoff of greater than 0.9
                    if new.iloc[i].astype("float")["{}_likelihood".format(limb)] > pcutoff:
                        y = new.iloc[i].astype('float')["{}_y".format(limb)]
                        x = new.iloc[i].astype('float')["{}_x".format(limb)]
                        if y > m*x + c + slipthresh_: #if the value of y falls below the rung line, then add to the list
                            bliX.append([x, y])
                number_of, centres = self.meanShiftClustering(bliX, slipthresh_)
                print("errors for {}: {}, and their centres: {}".format(limb, number_of, centres))
            traversal += 1
            run_v += 2

    def instancesBelowRungs(self, limb="BackLeft", pcutoff=0.99, plot="All", filter=6):
        run_number = ["one", "two", "three", "four", "five", "six"]
        animal_id = self.get_csv_filename().split("/")[-1][0:4]
        if limb not in self.limbs:
            return "This isn't a feature... typo?"
        else:
            #print("Results for {} limb.".format(limb))
            new = self.clean_data()
            slices = self.clean_slices(filter=filter)
            run_v = 0
            traversal = 1
            results = {}
            m, c = self.get_nose_line_equation()
            slipthresh_ = self.bodylength()/8
            for run in range(int(len(slices) / 2)):
                index = "{}_{}".format(run_number[traversal-1], animal_id)
                cumulativeError = 0
                limb_x = []
                limb_y = []
                bliX = []
                for i in range(slices[run_v], slices[run_v + 1]): #iterate through each run and return any coordinates that are below the rung lines with a pcutoff of greater than 0.9
                    if new.iloc[i].astype("float")["{}_likelihood".format(limb)] > pcutoff:
                        y = new.iloc[i].astype('float')["{}_y".format(limb)]
                        x = new.iloc[i].astype('float')["{}_x".format(limb)]
                        if y > m*x + c + slipthresh_: #if the value of y falls below the rung line, then add to the list
                            limb_x.append(x)
                            limb_y.append(y)
                            bliX.append([x, y])
                if plot == "All":
                    self.plot_rungs(limb_x, limb_y, plotSlip=True, slipthresh=slipthresh_) #plot the coordinates on the first frame along with rung line
                elif plot == traversal:
                    #print("Traversal: {}".format(traversal))
                    self.plot_rungs(limb_x, limb_y, plotSlip=True, slipthresh=slipthresh_) #plot the coordinates on the first frame along with rung line and slip threshold
                if limb == "FrontLeft":
                    results[index] = [self.meanShiftClustering(bliX, slipthresh_)[0], 1, 0, 0, 0]
                elif limb == "FrontRight":
                    results[index] = [self.meanShiftClustering(bliX, slipthresh_)[0], 0, 1, 0, 0]
                elif limb == "BackLeft":
                    results[index] = [self.meanShiftClustering(bliX, slipthresh_)[0], 0, 0, 1, 0]
                elif limb == "BackRight":
                    results[index] = [self.meanShiftClustering(bliX, slipthresh_)[0], 0, 0, 0, 1]
                traversal += 1
                run_v += 2
            results = pd.DataFrame.from_dict(results, orient='index', columns=['Clusters',
            'front left', 'front right', 'back left', 'back right'])
            return results
