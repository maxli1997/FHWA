import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import KFold
from sklearn import metrics
import os

# read events and data
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

# train individual model
for dr in driver_list:
    dr_df = event_df[event_df['driver']==dr]
    archive = []
    observations = []
    labels = []
    length = []
    outnames = []

    # create training samples
    for i,row in dr_df.iterrows():
        driver = row['driver']
        trip = row['trip']
        starttime = row['starttime']
        endtime = row['endtime']
        data = data_df[(data_df['Driver']==driver) & (data_df['Trip']==trip)]
        video_name = str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt'
        filename = 'GT_'+str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt'
        f = open('./res-cache/'+filename, 'r')
        contents = f.readlines()
        f.close()
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
        outnames.append('/HMM_'+str(driver).zfill(3)+'_'+str(trip).zfill(4)+'_'+str(starttime)+'.txt')
        truths = []
        if data.empty:
            continue
        for content in contents:
            content = content.strip('\n')
            if content=='True':
                truths.append(1)
            else:
                truths.append(0)
        labels.append(truths)
        observation = []
        i = 0
        for j,r in data.iterrows():
            if r['Time'] in range(starttime, endtime):
                if i==min_frame:
                    break
                p = int ((cabin_res[i]+face_res[i])>=1)
                observation.append([r['Speed'],r['Steer'],r['Ax'],r['LdwLateralSpeed'],p])
                i+=1

        length.append(len(observation))
        archive.append(observation)

    archive = np.array(archive)
    length = np.array(length)

    # train model
    kf = KFold(n_splits=3)
    accuracy = []
    f1 = []
    roc = []

    model = hmm.GaussianHMM(n_components=3, tol=0.00001, n_iter=100, verbose=True, covariance_type="full", init_params='cm')
    model.startprob_ = np.array([1.0, 0.0, 0.0])
    model.transmat_ = np.array([[0.5, 0.5,0.0],
                                [0.3, 0.5,0.2],
                                [0.4,0.2,0.4]])
    
    if len(archive) < 3:
        continue

    for train_index, test_index in kf.split(archive):
        X_train, X_test = archive[train_index], archive[test_index]
        train_length = length[train_index]
        y_test = []
        for i in test_index:
            y_test.append(labels[i])
        observations = np.concatenate(X_train)
        model.fit(observations,lengths=train_length)
    
        preds = []
        test_label = []
        for test,i in zip(X_test, y_test):
            result = model.predict(test)
            for r in result:
                if r==1:
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

    # after tuning the parameters, predict all events
    for i in range(5):
        if not os.path.exists('./hmm_single/hmm_results_'+str(i+1)):
            os.mkdir('./hmm_single/hmm_results_'+str(i+1))
        model.fit(np.concatenate(archive),lengths=length)
        for observation,filename in zip(archive,outnames):
            pred = model.predict(observation)
            f = open('./hmm_single/hmm_results_'+str(i+1)+filename, 'w')
            for p in pred:
                if p == 1:
                    f.write('True\n')
                else:
                    f.write('False\n')
            f.close()

print(np.mean(total_accuracy),np.mean(total_f1),np.mean(total_roc))
