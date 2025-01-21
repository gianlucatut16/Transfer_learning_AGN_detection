print('Setting everything up...')
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.utils.io import capture_output
import efficientnet_pytorch
import sys
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225]),
        ])

def normalize(images):
    min_val = np.min(images)
    max_val = np.max(images)
    range_val = max_val - min_val
    normalized_images = ((images - min_val) / range_val)
    return normalized_images

class NumpyDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = Image.fromarray(x.astype(np.uint8))
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.data)

@torch.no_grad()
def get_latent_vectors(network, train_loader, device):
    network.eval()
    latent_vectors = []
    for cnt, x in enumerate(train_loader):
        x = x.to(device) 
        latent_vectors.append(network.extract_features(x).mean(dim=(2,3)))
    latent_vectors = torch.cat(latent_vectors).cpu().numpy()
    return latent_vectors  

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        with capture_output():
            value = func(*args, **kwargs)
        return value
    return func_wrapper

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

@blockPrinting
def get_features(loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     blockPrint()
    network = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
#     enablePrint()    
    network.to(device)
    network.eval()
    features = get_latent_vectors(network, loader, device)
    return features

with open('complete_df.pkl', 'rb') as file:
    df = pickle.load(file)

for i in df.columns.values:
    for j in range(2675):
        try:
            if df[i][j].shape == (51, 51, 1):
                df[i][j] = df[i][j][:,:,0]
        except:
            print('ERROR')
            print(i, j)

def norm_imgs(column):
    size = 51
    test_imgs = df[column]
    images = np.zeros((len(test_imgs), size , size,3), dtype= np.float32)

    for i in range(len(test_imgs)):
        img = test_imgs[i]
        img = Image.fromarray(img).convert('RGB')
        img= np.array(img)          
        images[i]=img

    return images

FEATS = ['Freq1_harmonics_amplitude_0', 'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2', 'Freq1_harmonics_amplitude_3',
'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1', 'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3',
'Freq3_harmonics_amplitude_0', 'Freq3_harmonics_amplitude_1', 'Freq3_harmonics_amplitude_2', 'Freq3_harmonics_amplitude_3',
'Freq1_harmonics_rel_phase_0', 'Freq1_harmonics_rel_phase_1', 'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3',
'Freq2_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_3',
'Freq3_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_1', 'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3',
'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Amplitude', 'AndersonDarling', 'Autocor_length', 'Con', 'Eta_e',
'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35', 'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
'FluxPercentileRatioMid80', 'Gskew', 'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP',
'PairSlopeTrend', 'PercentAmplitude', 'PercentDifferenceFluxPercentile', 'Q31', 'Rcs', 'Skew', 'SmallKurtosis', 'Std',
'StructureFunction_index_21', 'StructureFunction_index_31', 'StructureFunction_index_32', 'Median']

importances = pd.DataFrame(data = None, columns = ['feature', 'accuracy', 'recall'])

for column in FEATS:
    print(f'*************** Processing {column} *****************')
    print('Normalize images...')
    images = norm_imgs()
    dataset = NumpyDataset(images, transform = transform)


    if "linux" in sys.platform:
        nw=torch.get_num_threads()-1
    else:
        nw=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=nw)

    print('Get features for Nearest Neighbor...')
    X = get_features(loader)
    y = np.array(df['AGN'].astype(int))

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, shuffle=True, random_state=2)

    print('Runing Nearest Neighbors...')
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy, recall = accuracy_score(y_test, y_pred), recall_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
    plt.savefig(f'Heatmap_{column}.png')
    importances[len(importances)] = [column, accuracy, recall]

importances.to_csv('importances.csv', index = False)


sns.barplot(data = importances.sort_values(by = 'accuracy', ascending = False), x = 'accuracy', y = 'column')
plt.title('Accuracy importances')
plt.xlabel('Accuracy')
plt.ylabel('Features')
plt.savefig('Accuracy_importances.png')

sns.barplot(data = importances.sort_values(by = 'recall', ascending = False), x = 'recall', y = 'column')
plt.title('Recall importances')
plt.xlabel('Recall')
plt.ylabel('Features')
plt.savefig('Recall_importances.png')