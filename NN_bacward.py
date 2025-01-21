import os
import sys
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from PIL import Image
from IPython.utils.io import capture_output

import efficientnet_pytorch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.neighbors import NearestNeighbors

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_nns(query, neigh):
    res = neigh.kneighbors(query)
    similar = res[1][0]
    dists = res[0][0]
    return similar, dists


def get_neighbors_and_clf(values: list, n_neighbors:int, tresh_agn:int):
    neighbors = values[1 : n_neighbors + 1]
    neighbors_target = [df_target['AGN'][value] for value in neighbors]

    agn_count, noagn_count = 0, 0
    for value in neighbors_target:
        if value == True:
            agn_count += 1
        else:
            noagn_count += 1

    prob_agn = agn_count / n_neighbors
    prob_noagn = noagn_count / n_neighbors 

    if prob_agn >= tresh_agn:
        clf = True
    else:
        clf= False
    
    return clf

F1_AGN_0 = 0.4580801944106926

with open('final_df.pkl', 'rb') as file:
    df_complete = pickle.load(file)


bw_sel = ['Q31', 'Freq3_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1',
          'Amplitude', 'MedianAbsDev', 'Freq3_harmonics_amplitude_3', 'AndersonDarling', 'PeriodLS', 'Freq1_harmonics_amplitude_2',
          'Freq2_harmonics_amplitude_1', 'SmallKurtosis', 'Freq2_harmonics_amplitude_3', 'Freq1_harmonics_amplitude_1', 'Std',
          'Freq1_harmonics_amplitude_3', 'Median', 'Freq1_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_0', ]

for j in bw_sel:
    columns_to_drop = [col for col in df_complete.columns if ('_'.join(col.split('_')[:-1]) == j)]
    df_complete = df_complete.drop(columns=columns_to_drop)

with open('true_df_21.pkl', 'rb') as file:
    df_target = pickle.load(file)

cols = list(df_complete.columns.values)
feats = np.unique(['_'.join(x.split('_')[:-1]) for x in cols])
print(feats, len(feats))


results_df = pd.DataFrame(data = None, columns = ['drop_column', 'recall_agn', 'precision_agn', 'f1_agn', 'recall_noagn', 'precision_noagn', 'f1_noagn', 'diff'])

counter = 1
for feat in feats:

    columns_to_drop = [col for col in df_complete.columns if ('_'.join(col.split('_')[:-1]) == feat)]
    df1 = df_complete.drop(columns=columns_to_drop)
    print(f'********************* {feat} {df1.shape} {counter}/{len(feats)} *********************')
    df = df1.values

    
    print('Running Nearest Neighbors...')
    neigh = NearestNeighbors(n_neighbors=30)
    neigh.fit(df)

    print('Taking votes...')
    preds = []
    for indice in range(df.shape[0]):
        query_features = df[indice].reshape(1, -1)
        similar, dists = get_nns(query_features, neigh)
        preds.append(similar)
        if indice % 200 == 0:
            print(indice)

    print('Appending results to the dataset')
    df1['neighbors'] = preds

    df1['clf'] = df1['neighbors'].apply(get_neighbors_and_clf, n_neighbors = 23, tresh_agn = 0.3)

    # Recall metric
    recall_per_class = recall_score(df_target['AGN'], df1['clf'], average=None)
    recall_noagn = recall_per_class[0]
    recall_agn = recall_per_class[1]

    # Precision metric
    precision_per_class = precision_score(df_target['AGN'], df1['clf'], average=None)
    precision_noagn = precision_per_class[0]
    precision_agn = precision_per_class[1]

    # F1 - score metric
    f1_scores = f1_score(df_target['AGN'], df1['clf'], average=None)
    f1score_noagn = f1_scores[0]
    f1score_agn = f1_scores[1]


    diff_f1 = F1_AGN_0 - f1score_agn

    results_df.loc[len(results_df)] = [feat, recall_agn, precision_agn, f1score_agn, recall_noagn, precision_noagn, f1score_noagn, diff_f1]

    if counter % 3 == 0:
        results_df.to_csv('risultati_backward_21.csv', index = False)

    counter += 1

results_df = results_df.sort_values('diff')
results_df.to_csv('risultati_backward_21.csv', index = False)

