import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    f = open('label.txt', 'r')
    contents = f.readlines()
    f.close()

    total_map = {}
    frame_map = {}
    video_list = []
    cabin_list = []

    for line in contents:
        content_list = line.strip().split(' ')
        if content_list[4] not in video_list:
            video_list.append(content_list[4])
            cabin_list.append(content_list[5])
        if content_list[4] in total_map:
            frame_map[content_list[4]].append([int(content_list[2]), int(content_list[3])])
        else:
            total_map[content_list[4]] = int(content_list[6])
            frame_map[content_list[4]] = [[int(content_list[2]), int(content_list[3])]]
    t_acc = []
    t_f1_s = []
    t_pre = []
    t_rec = []


    for k in range(5):
        acc = []
        f1_s = []
        pre = []
        rec = []
        out = open('accuracy.txt', 'w')
        for idx, video in enumerate(video_list):
            str_list = video.strip().split('.')
            try:
                f1 = open("./res-cache/GT" + str_list[0][4:] + ".txt")
                f2 = open("./res-cache/Face" + str_list[0][4:] + ".txt")
                f3 = open("./res-cache/Cabin" + str_list[0][4:] + ".txt")
                f4 = open("./hmm_single/hmm_results_"+str(k+1)+"/HMM" + str_list[0][4:] + ".txt")
                f5 = open("./svc_single/svc_results_"+str(k+1)+"/SVC" + str_list[0][4:] + ".txt")
                f6 = open("./lstm_single/lstm_results/LSTM" + str_list[0][4:] + ".txt")
            except:
                print("read file fail of video", video)
                continue

            cabin_res = []
            for ele in f3.readlines():
                for _ in range(5):
                    if ele.strip() == "True":
                        cabin_res.append(True)
                    else:
                        cabin_res.append(False)
            
            gt_res = []
            for ele in f1.readlines():
                if ele.strip() == "True":
                    gt_res.append(True)
                else:
                    gt_res.append(False)

            face_res = []
            for ele in f2.readlines():
                if ele.strip() == "True":
                    face_res.append(True)
                else:
                    face_res.append(False)
            
            hmm_res = []
            for ele in f4.readlines():
                if ele.strip() == "True":
                    hmm_res.append(True)
                else:
                    hmm_res.append(False)

            svc_res = []
            for ele in f5.readlines():
                if ele.strip() == "True":
                    svc_res.append(True)
                else:
                    svc_res.append(False)

            lstm_res = []
            for ele in f6.readlines():
                if ele.strip() == "True":
                    lstm_res.append(True)
                else:
                    lstm_res.append(False)

            total_res = face_res
            for idx in range(min(len(cabin_res), len(face_res))):
                a = int (face_res[idx] == True)
                b = int (cabin_res[idx] == True)
                c = int (hmm_res[idx] == True)
                d = int (svc_res[idx] == True)
                e = int (lstm_res[idx] == True)

                # choose which model a+b c d e
                if e>=1:
                    total_res[idx] = True
                else:
                    total_res[idx] = False
            
            correct = 0
            for i in range(len(total_res)):
                if gt_res[i] == total_res[i]:
                    correct += 1
            
            if len(total_res)<len(gt_res):
                gt_res = gt_res[:len(total_res)]
            
            f1_s.append(metrics.f1_score(gt_res,total_res))
            pre.append(metrics.precision_score(gt_res,total_res))
            rec.append(metrics.recall_score(gt_res,total_res))
            acc.append(metrics.accuracy_score(gt_res,total_res))
            # print("video:", video)
            # print("accuracy:", float(correct) / float(len(total_res)) * 100.0)
            # print("count:", correct, len(total_res))
            #out.write("video: " + video + "\n")
            #out.write("accuracy: " + str(float(correct) / float(len(total_res)) * 100.0) + "\n")
            #out.write("count: " + str(correct) + " " + str(len(total_res)) + "\n")
            #acc.append(float(correct) / float(len(total_res)) * 100.0)
        t_acc.append(np.mean(acc))
        t_f1_s.append(np.mean(f1_s))
        t_pre.append(np.mean(pre))
        t_rec.append(np.mean(rec))

    print(np.mean(t_acc),np.mean(t_f1_s),np.mean(t_pre),np.mean(t_rec))