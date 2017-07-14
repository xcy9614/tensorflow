import json
import csv

action_info = []
voice_info = []
info = []
def ReadActionInfo(file_name): 
    file = open('res/'+file_name)
    lines = file.readlines()
    action_info_index = len(action_info)-1
    for i in range(len(lines)):
        if i % 3 == 0:
            # 确定帧数
            action_info_index += 1
            action_info.append([])
        elif i % 3 == 1:
            # 存入中心点坐标，即每一帧的前三位
            lines[i] = lines[i].replace('\n','')
            s = json.loads(lines[i])
            for coord in s:
                action_info[action_info_index].append(coord)
        else:
            # 存入每个关节的坐标
            lines[i] = lines[i].replace('\n','')
            s = json.loads(lines[i])
            for joint in s:
                for coord in joint:
                    action_info[action_info_index].append(coord)

    file.close()

def ReadVoiceInfo(file_name):
    file = open('data1/'+file_name)
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n','')
        s = json.loads(line)
        voice_info.append(s)
    file.close()

def WriteCsv():
    for i in range(len(action_info)):
        info.append(voice_info[i]+action_info[i])
    # print(info[0])
    csvfile = open('train.csv','w',newline='')
    mywriter = csv.writer(csvfile,dialect='excel')
    mywriter.writerows(info)
    csvfile.close()

for i in range(6):
    ReadActionInfo('erdongzuo'+str(i))
    ReadVoiceInfo(str(i)+'.txt')
# ReadActionInfo('erdongzuo1')
# ReadVoiceInfo('1.txt')
WriteCsv()
