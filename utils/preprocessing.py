import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Separate the dataset into two parts based on 'fraud' column: fraudulent and non-fraudulent
    fraudulent_data = df[df['fraud'] == 1]
    non_fraudulent_data = df[df['fraud'] == 0]

    # Randomly select 25,000 tuples from each class
    sample_size_for_fraudulent = 17000
    fraudulent_sample = fraudulent_data.sample(n=sample_size_for_fraudulent, random_state=42)
    non_fraudulent_sample = non_fraudulent_data.sample(n=(150000 - sample_size_for_fraudulent), random_state=42)

    # Concatenate the two samples to get the final random subset of 25,000 tuples
    random_sample = pd.concat([fraudulent_sample, non_fraudulent_sample], axis=0)

    # Shuffle the subset to ensure the tuples are randomly arranged
    random_sample = random_sample.sample(frac=1, random_state=42)

    # 'random_sample' contains our stratified random subset of 25,000 tuples
    df=random_sample
    #Scaling numerical features
    #the numercal features that need to be scaled are  'distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price'
    scaler=MinMaxScaler()
    scaler.fit(df[['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']])
    df[['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']] = scaler.transform(df[['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price']])


    #separate the features and the labels

    return df


