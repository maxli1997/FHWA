import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import metrics
from sklearn.model_selection import KFold
from sklearn import metrics
import os

# choose sample length and prediction gap
timestep = 15
leadtime = 5

# read data
event_df = pd.read_csv('./fhwa_events.csv')
data_df = pd.read_csv('./fhwa_data.csv')
data_df = data_df.drop_duplicates()

# normalize data
norm_cols = ['Speed','Steer','LdwLateralSpeed','Ax','Ay']
normalized_df=(data_df[norm_cols]-data_df[norm_cols].min())/(data_df[norm_cols].max()-data_df[norm_cols].min())
data_df[norm_cols] = normalized_df

driver_list = event_df.driver.unique()
total_accuracy = []
total_f1 = []
total_roc = []

# train individual data
for dr in driver_list:
    dr_df = event_df[event_df['driver']==dr]
    archive = []
    observations = []
    labels = []
    true_labels = []
    length = []
    outnames = []

    for i,row in dr_df.iterrows():
        driver = row['driver']
        trip = row['trip']
        starttime = row['starttime']
        endtime = row['endtime']
        data = data_df[(data_df['Driver']==driver) & (data_df['Trip']==trip)]
        filename = 'GT_'+str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt'
        f = open('./res-cache/'+filename, 'r')
        contents = f.readlines()
        f.close()
        outnames.append('/LSTM_'+str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt')
        truths = []
        if data.empty:
            print('No available trip')
            continue
        for content in contents:
            content = content.strip('\n')
            if content=='True':
                truths.append(True)
            else:
                truths.append(False)
        labels.append(truths)

        observation = []
        data = data.to_numpy()
        for i in range(0,len(data)-(timestep+leadtime-1)):
            if data[i][2] not in range(starttime, endtime):
                continue
            if data[i+(timestep+leadtime-1)][2] not in range(starttime, endtime):
                continue
            t =[]
            for j in range(i,i+timestep):
                u = []
                for k in range(3,7):
                    u.append(data[j][k])
                t.append(u)
            observation.append(t)
        
        labelset = truths[(timestep+leadtime-1):(timestep+leadtime-1)+len(observation)]

        length.append(len(observation))
        archive.append(observation)
        true_labels.append(labelset)

    archive = np.array(archive)
    length = np.array(length)
    true_labels = np.array(true_labels)

    # tune model    
    kf = KFold(n_splits=3)
    accuracy = []
    f1 = []
    roc = []
    
    if len(archive) < 3:
        continue

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timestep, 4)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    for train_index, test_index in kf.split(archive):
        X_train, X_test = archive[train_index], archive[test_index]
        Y_train, Y_test = true_labels[train_index], true_labels[test_index]
        train_length = length[train_index]
        observations = np.concatenate(X_train)
        l = np.concatenate(Y_train)
        model.fit(observations, l, epochs = 25, batch_size = 64)
    
        preds = []
        test_label = []
        for test,i in zip(X_test, Y_test):
            result = model.predict(test)
            for r in result:
                if r>0.5:
                    preds.append(True)
                else:
                    preds.append(False)
            if len(test) < len(i):
                for j in range(len(i)-len(test)):
                    preds.append(False)
            for k in i:
                test_label.append(k)
        preds = np.array(preds)

        accuracy.append(metrics.accuracy_score(test_label,preds))
        f1.append(metrics.f1_score(test_label,preds,pos_label=True))  
        roc.append(metrics.roc_auc_score(test_label,preds))
    total_accuracy.append(np.mean(accuracy))
    total_f1.append(np.mean(f1))
    total_roc.append(np.mean(roc))

    # predict after tuning
    for i in range(5):
        if not os.path.exists('./lstm_single/lstm_results_'+str(i+1)):
            os.mkdir('./lstm_single/lstm_results_'+str(i+1))
        model.fit(np.concatenate(archive),np.concatenate(true_labels), epochs = 50, batch_size = 64)
        for observation,filename in zip(archive,outnames):
            pred = model.predict(observation)
            f = open('./lstm_single/lstm_results_'+str(i+1)+filename, 'w')
            for j in range((timestep+leadtime-1)):
                f.write('False\n')
            for p in pred:
                if p > 0.5:
                    f.write('True\n')
                else:
                    f.write('False\n')
            f.close()

print(np.mean(total_accuracy),np.mean(total_f1),np.mean(total_roc))