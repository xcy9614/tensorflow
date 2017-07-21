import json
import csv
from numpy import *
from musigma import MUSIGMA_V,MUSIGMA_A

sum_action_info = []
sum_voice_info = []
fin_action_info = []
fin_voice_info = []
beat_info = []
info = []
max_length = 73

def ReadActionInfo(file_name): 
    action_info = []
    file = open('res/'+file_name)
    lines = file.readlines()
    action_info_index = -1
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
    sum_action_info.append(action_info)

def ReadVoiceInfo(file_name): 
    voice_info = []
    file = open('data1/'+file_name)
    lines = file.readlines()
    # print(len(lines))
    for line in lines:
        line = line.replace('\n','')
        s = json.loads(line)
        voice_info.append(s)
    file.close()
    sum_voice_info.append(voice_info)
    # print('voice___info: ' + str(matrix(sum_voice_info).shape))

def ReadBeatInfo(file_name):
    file = open('beats/'+file_name)
    lines = file.readlines()
    beat_num = []
    for line in lines:
        line = line.replace('\n','')
        beat_num.append(int(float(line)))
    beat_info.append(beat_num)
    file.close()



def WriteCsv():
    for i in range(len(fin_action_info)):
        info.append(fin_voice_info[i]+fin_action_info[i])
    print('info: ' + str(matrix(info).shape))
    csvfile = open('train.csv','w',newline='')
    mywriter = csv.writer(csvfile,dialect='excel')
    mywriter.writerows(info)
    csvfile.close()

for i in range(23):
    ReadActionInfo(str(i+1))
    ReadVoiceInfo(str(i+1)+'.txt')
    ReadBeatInfo(str(i+1)+'_beat.txt')

for k in range(len(sum_voice_info)):
    N=len(sum_voice_info[k])
    N1=len(sum_voice_info[k][0])
    matrix_voice=array(sum_voice_info[k])
    for i in range(0,N1):
        for j in range(0,N):
            matrix_voice[j,i]= (matrix_voice[j,i] - MUSIGMA_V[i][0]) / MUSIGMA_V[i][1];  
    sum_voice_info[k]=matrix_voice.tolist()

for k in range(len(sum_action_info)):
    M=len(sum_action_info[k])
    M1=len(sum_action_info[k][0])
    matrix_action=array(sum_action_info[k])
    for i in range(0,M1):
        for j in range(0,M):
            matrix_action[j,i]= (matrix_action[j,i] - MUSIGMA_A[i][0]) / MUSIGMA_A[i][1];  
    sum_action_info[k]=matrix_action.tolist()

for i in range(len(beat_info)):
    for j in range(len(beat_info[i])-1):
        start_num = len(fin_voice_info)
        fin_action_info += sum_action_info[i][beat_info[i][j]:beat_info[i][j+1]]
        fin_voice_info += sum_voice_info[i][beat_info[i][j]:beat_info[i][j+1]]

        for k in range(max_length-(beat_info[i][j+1]-beat_info[i][j])):
            action_zero = [0 for i in range(66)]
            voice_zero = [0 for i in range(19)]
            fin_action_info.append(action_zero)
            fin_voice_info.append(voice_zero)
        # 加入在动作中的帧号
        for l in range(max_length):
            fin_voice_info[start_num+l].append(l)

# 加入在总体中的帧号
for i in range(len(fin_voice_info)):
    fin_voice_info[i].append(i)

for i in range(len(fin_voice_info)):
    if len(fin_voice_info[i]) != 21:
        print(len(fin_voice_info[i]))
print('final voice info: ' + str(matrix(fin_voice_info).shape))
print('final action info: ' + str(matrix(fin_action_info).shape))


N=len(fin_voice_info)
N1=len(fin_voice_info[0])
matrix_voice=array(fin_voice_info)
for i in range(19,N1):
    mu=average(matrix_voice[:,i])        
    sigma=std(matrix_voice[:,i])
    for j in range(0,N):
        matrix_voice[j,i] = (matrix_voice[j,i] - mu) / sigma;  
fin_voice_info=matrix_voice.tolist()

WriteCsv()
