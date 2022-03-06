import xlrd
import cv2
import os
import sys
 
workbook = xlrd.open_workbook("./ivbss/cell_phone_review_codes.xls")
worksheet = workbook.sheet_by_index(0)

nrows = worksheet.nrows

possible_events = []
enter = 1

for i in range(nrows):
    if enter == 1:
        enter = 0
        continue    
        
    possible_events.append(worksheet.row_values(i))


last_num = -1.0
events = []
driver = 0.0
trip = 0.0
begin = 0.0
for ele in possible_events:
    if last_num == -1.0:
        driver = ele[0]
        trip = ele[1]
        begin = ele[2]
        last_num = ele[3]
    elif int(last_num) % 2 == 1:
        if int(ele[3]) == int(last_num) + 1:
            if driver == ele[0] and trip == ele[1]:
                events.append([driver, trip, begin, ele[2]])
                last_num = -1.0
                driver = 0.0
                trip = 0.0
                begin = 0.0
        else:
            driver = ele[0]
            trip = ele[1]
            begin = ele[2]
            last_num = ele[3]
    else:
        last_num = -1.0

# print(len(events))

available_videos = []
all_videos = os.listdir('ivbss/phone')
all_videos.sort()
for file_name in all_videos:
    if file_name.endswith('.avi') and file_name.startswith('Face_'):
        cap = cv2.VideoCapture('ivbss/phone/' + file_name)

        file_name_split = file_name.split('.')
        part_split = file_name_split[0].split('_')
        begin = float(part_split[-1].strip())

        if cap.isOpened(): 
            frameNumber = cap.get(7)

        # print(file_name, part_split[1], part_split[2], begin, frameNumber * 10 + begin)
        available_videos.append([file_name, part_split[1], part_split[2], begin, frameNumber * 10 + begin, frameNumber])

# for ele in available_videos:
#     print(ele)
whole_events = []

for ele in events:
    file_path_cabin = 'ivbss/phone/Cabin_'
    file_path_face = 'ivbss/phone/Face_'
    driver_str = str(int(ele[0]))
    driver_len = len(str(int(ele[0])))
    for i in range(3 - driver_len):
        driver_str = '0' + driver_str
    trip_str = str(int(ele[1]))
    trip_len = len(str(int(ele[1])))
    for i in range(4 - trip_len):
        trip_str = '0' + trip_str
    
    file_path_cabin += driver_str + '_' + trip_str + '_'
    file_path_face += driver_str + '_' + trip_str + '_'

    for avail in available_videos:
        if avail[1] == driver_str and avail[2] == trip_str:
            if ele[2] >= avail[3] and ele[3] <= avail[4]:
                face_video = avail[0]
                cabin_video = 'Cabin' + avail[0][4:]
                whole_events.append([ele[0], ele[1], (ele[2] - avail[3]) / 10, (ele[3] - avail[3]) / 10, face_video, cabin_video, avail[5]])

f = open('label.txt','w')
for ele in whole_events:
    f.write(str(int(ele[0]))+' '+str(int(ele[1]))+' '+str(int(ele[2]))+' '+str(int(ele[3]))+' '+str(ele[4])+' '+str(ele[5])+' '+str(int(ele[6]))+'\n')
f.close()
