from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import os

# choose sample length and prediction gap
timestep = 5
leadtime = 0

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
        
        # detailed comments in lstm_single.py
        try:
            f2 = open("./res-cache/Face_" + video_name)
            f3 = open("./res-cache/Cabin_" + video_name)
        except:
            print("read file fail of video", video_name)
            continue
        face_res = []
        for ele in f2.readlines():
            if ele.strip() == "True":
                face_res.append(True)
            else:
                face_res.append(False)
        f2.close()
        cabin_res = []
        for ele in f3.readlines():
            for _ in range(5):
                if ele.strip() == "True":
                    cabin_res.append(1)
                else:
                    cabin_res.append(0)
        f2.close()
        min_frame = min(len(cabin_res),len(face_res))
        if min_frame == 0:
            continue
        # ends here
        
        outnames.append('/SVC_'+str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt')
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
            for j in range(i,i+(timestep+leadtime)):
                for k in range(3,7):
                    t.append(data[j][k])
                    
                # detailed comments in lstm_single.py    
                try:
                    f = face_res[j]
                except:
                    f = 0
                try:
                    c =cabin_res[j]
                except:
                    c = 0
                if f>0 or c>0:
                    t.append(1)
                else:
                    t.append(0)
                # ends here
                
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
    
    for train_index, test_index in kf.split(archive):
        X_train, X_test = archive[train_index], archive[test_index]
        Y_train, Y_test = true_labels[train_index], true_labels[test_index]
        train_length = length[train_index]
        observations = np.concatenate(X_train)
        l = np.concatenate(Y_train)
        clf = svm.SVC()
        clf.fit(observations,l)
    
        preds = []
        test_label = []
        for test,i in zip(X_test, Y_test):
            result = clf.predict(test)
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
        if not os.path.exists('./svc_single/svc_results_'+str(i+1)):
                os.mkdir('./svc_single/svc_results_'+str(i+1))
        clf.fit(np.concatenate(archive),np.concatenate(true_labels))
        for observation,filename in zip(archive,outnames):
            pred = clf.predict(observation)
            f = open('./svc_single/svc_results_'+str(i+1)+filename, 'w')
            for j in range((timestep+leadtime-1)):
                f.write('False\n')
            for p in pred:
                if p > 0.5:
                    f.write('True\n')
                else:
                    f.write('False\n')
            f.close()

print(np.mean(total_accuracy),np.mean(total_f1),np.mean(total_roc))
