import subprocess
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import sys

def generate_gt(video_name,total_frame,start_end_frame):
    ground_truth = []
    for i in range(total_frame):
        ground_truth.append(False)

    for temp_frame in start_end_frame:
        for i in range(temp_frame[0], temp_frame[1]):
            ground_truth[i] = True

    f = open("./res-cache/" + "GT" + video_name[4:] + ".txt", "w")
    for ele in ground_truth:
        f.write(str(ele) + "\n")
    f.close()
    return

def extract_angle(video_name, total_frame, start_end_frame):
    try:
        f = open('./OpenFace/build/processed/' + video_name + '.csv', 'r')
    except:
        print("error: can't open")
        return False

    contents = f.readlines()

    header = contents[0].strip().split(',')

    eye_0_idx = []
    eye_1_idx = []
    head_idx = []

    for idx, ele in enumerate(header):
        if ele == 'gaze_0_x' or ele == 'gaze_0_y' or ele == 'gaze_0_z':
            eye_0_idx.append(idx)
        if ele == 'gaze_1_x' or ele == 'gaze_1_y' or ele == 'gaze_1_z':
            eye_1_idx.append(idx)
        if ele == 'pose_Tx' or ele == 'pose_Ty' or ele == 'pose_Tz' or ele == 'pose_Rx' or ele == 'pose_Ry' or ele == 'pose_Rz':
            head_idx.append(idx)

    enter = 1

    eye_0 = []
    eye_1 = []
    head = []
    frames = []

    for line in contents:
        if enter == 1:
            enter = 0
            continue

        line_content = line.strip().split(',')
        eye_0.append([float(line_content[e]) for e in eye_0_idx])
        eye_1.append([float(line_content[e]) for e in eye_1_idx])
        head.append([float(line_content[e]) for e in head_idx])
        frames.append(int(line_content[0]) - 1)
        
    trans_mat = []
    trans_eye_0 = []
    trans_eye_1 = []

    for ele in head:
        trans = np.eye(4)
        pitch, yaw, roll = ele[3], ele[4], ele[5]
        print("pyr:", pitch, yaw, roll)

        pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
        ])

        rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
        ])

        trans[0:3, 0:3] = pitchMatrix.dot(yawMatrix.dot(rollMatrix))
        # trans[0:3, 3] = ele[0:3]
        # print(trans)
        trans_mat.append(np.linalg.inv(trans))

    count = 0
    res = []

    for idx in range(len(trans_mat)):
        homo_0 = np.array([[eye_0[idx][0]], [eye_0[idx][1]], [eye_0[idx][2]], [1]])
        # print("eye_0:", eye_0[idx])
        homo_1 = np.array([[eye_1[idx][0]], [eye_1[idx][1]], [eye_1[idx][2]], [1]])
        # print("eye_1:", eye_1[idx])
        trans_0 = np.dot(trans_mat[idx], homo_0)
        trans_1 = np.dot(trans_mat[idx], homo_1)
        gaze_direction0 = [trans_0[0][0], trans_0[1][0], trans_0[2][0]]
        gaze_direction1 = [trans_1[0][0], trans_1[1][0], trans_1[2][0]]
        trans_eye_0.append(gaze_direction0)
        trans_eye_1.append(gaze_direction1)

        x_angle_0 = np.arctan2(gaze_direction0[0], -gaze_direction0[2])
        y_angle_0 = np.arctan2(gaze_direction0[1], -gaze_direction0[2])

        x_angle_1 = np.arctan2(gaze_direction1[0], -gaze_direction1[2])
        y_angle_1 = np.arctan2(gaze_direction1[1], -gaze_direction1[2])

        if max(y_angle_0, y_angle_1) > 0:
            res.append(True)
            count += 1
        else:
            res.append(False)
        
    ground_truth = []
    test_res = []
    for i in range(total_frame):
        ground_truth.append(False)
        test_res.append(False)

    for temp_frame in start_end_frame:
        for i in range(temp_frame[0], temp_frame[1]):
            ground_truth[i] = True

    for idx, e in enumerate(frames):
        test_res[e] = res[idx]

    # correct = 0
    # for i in range(total_frame):
    #     if ground_truth[i] == test_res[i]:
    #         correct += 1

    # print("video:", video_name)
    # print("accuracy:", float(correct) / float(total_frame) * 100.0)
    # print("count:", correct, total_frame)
    f = open("./res-cache/" + video_name + ".txt", "w")
    for ele in test_res:
        f.write(str(ele) + "\n")
    f.close()

    f = open("./res-cache/" + "GT" + video_name[4:] + ".txt", "w")
    for ele in ground_truth:
        f.write(str(ele) + "\n")
    f.close()
    return True

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


    video_count = 0
    out = open('accuracy.txt', 'w')

    for idx, video in enumerate(video_list):
        str_list = video.strip().split('.')
        generate_gt(str_list[0],total_map[video],frame_map[video])

    for idx, video in enumerate(video_list):
        video_count += 1
        subprocess.call(["./bin/FaceLandmarkVidMulti", "-f",  "../../IVBSS/phone/" + video], cwd="./OpenFace/build/")
        str_list = video.strip().split('.')
        correct_sign = extract_angle(str_list[0], total_map[video], frame_map[video])

        if correct_sign == False:
            continue

        subprocess.call(["python3", "detectvideo.py", cabin_list[idx]], cwd="./tensorflow-yolov4-tflite/")

        try:
            f1 = open("./res-cache/GT" + str_list[0][4:] + ".txt")
            f2 = open("./res-cache/Face" + str_list[0][4:] + ".txt")
            f3 = open("./res-cache/Cabin" + str_list[0][4:] + ".txt")
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

        total_res = face_res
        for idx in range(min(len(cabin_res), len(face_res))):
            total_res[idx] = total_res[idx] or cabin_res[idx]
        
        correct = 0
        for i in range(len(total_res)):
            if gt_res[i] == total_res[i]:
                correct += 1

        # print("video:", video)
        # print("accuracy:", float(correct) / float(len(total_res)) * 100.0)
        # print("count:", correct, len(total_res))
        out.write("video: " + video + "\n")
        out.write("accuracy: " + str(float(correct) / float(len(total_res)) * 100.0) + "\n")
        out.write("count: " + str(correct) + " " + str(len(total_res)) + "\n")

        if video_count >= 1:
            break

    out.close()
