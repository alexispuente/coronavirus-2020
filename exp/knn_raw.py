"""
Experiment summary
------------------
Treat each province/state in a country cases over time
as a vector, do a simple K-Nearest Neighbor between 
countries. What country has the most similar trajectory
to a given country?
"""

import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
from sklearn.neighbors import (
    KNeighborsClassifier,
    DistanceMetric
)
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#count the most frequent element in a link
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

#Create a list
nearest_neighbor = []

# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
N_NEIGHBORS = 15 #CHANGE THIS EVERYTIME
MIN_CASES = 1000
NORMALIZE = True
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_global.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []

for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    features.append(cases)
    targets.append(labels)

features = np.concatenate(features, axis=0)
targets = np.concatenate(targets, axis=0)
predictions = {}

for _dist in ['minkowski', 'manhattan']:
    counter = 0
    for val in np.unique(confirmed["Country/Region"]):
        counter = counter + 1
        # test data
        df = data.filter_by_attribute(
            confirmed, "Country/Region", val)
        cases, labels = data.get_cases_chronologically(df)

        # filter the rest of the data to get rid of the country we are
        # trying to predict
        mask = targets[:, 1] != val
        tr_features = features[mask]
        tr_targets = targets[mask][:, 1]

        above_min_cases = tr_features.sum(axis=-1) > MIN_CASES
        tr_features = tr_features[above_min_cases]
        if NORMALIZE:
            tr_features = tr_features / tr_features.sum(axis=-1, keepdims=True)
        tr_targets = tr_targets[above_min_cases]

        # train knn
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric=_dist)
        knn.fit(tr_features, tr_targets)

        # predict
        cases = cases.sum(axis=0, keepdims=True)
        # nearest country to this one based on trajectory
        label = knn.predict(cases)

        #add the label to the nearest neighbor list
        nearest_neighbor.append(label)
        """
        if val == "US":
            cases_us = cases.sum(axis=0)
            plt.plot(cases_us, label=labels[0,1])
            plt.legend()
            plt.savefig('results/US_trial.png')
        """
        
        if val not in predictions:
            predictions[val] = {}
        predictions[val][_dist] = label.tolist()
    #print("counter: ", counter)

most_frequent_n = most_frequent(nearest_neighbor)

##########################################################################################################################

#Get cases for the US
df = data.filter_by_attribute(
    confirmed, "Country/Region", "US")

cases, labels = data.get_cases_chronologically(df)

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#Plotting the US
plt.figure(0)
plt.title("US and Best Neighbor\n Overall Trajectories") #CHANGE TITLE
cases_us = cases.sum(axis=0)
plt.plot(cases_us, label=labels[0,1])

#Get cases for nearest neighbor
manhattan_neighbor = predictions["US"]['minkowski'] #CHANGE TO MINKOWSKI

df_neigh = data.filter_by_attribute(
    confirmed, "Country/Region", manhattan_neighbor[0])

cases_neigh, labels_neigh = data.get_cases_chronologically(df_neigh)

#plotting nearest neighbor
cases_new = cases_neigh.sum(axis=0)
plt.plot(cases_new, label=labels_neigh[0,1])

plt.xlabel("Time (days since Jan 22, 2020)")
plt.ylabel('# of confirmed cases')

plt.legend()
plt.savefig('US_predictions/N15_raw_minkowski.png')



with open('results/knn_raw.json', 'w') as f:
    json.dump(predictions, f, indent=4)
