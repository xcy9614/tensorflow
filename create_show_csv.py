# -*- coding: utf-8 -*-
import json
import csv
from numpy import *
from musigma import MUSIGMA_V,MUSIGMA_A
from lstm import TIME_STEPS,BATCH_SIZE,INPUT_SIZE,OUTPUT_SIZE
sum_voice_info = []
fin_voice_info = []
beat_info = []
max_length = 73

def ReadVoiceInfo(file_name): 
    voice_info = []
    file = open('music/'+file_name)
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
    file = open('music/'+file_name)
    lines = file.readlines()
    beat_num = []
    for line in lines:
        line = line.replace('\n','')
        beat_num.append(int(float(line)))
    beat_info.append(beat_num)
    file.close()

def WriteCsv():
    info=[]
    for i in range(len(fin_voice_info)):
        if fin_voice_info[i][0]==0 and fin_voice_info[i][1]==0:
            continue
        info.append(fin_voice_info[i])
    leng=matrix(info).shape[0]
    need=TIME_STEPS*BATCH_SIZE-leng%(TIME_STEPS*BATCH_SIZE)
    print(TIME_STEPS,BATCH_SIZE,need,matrix(info).shape[0],matrix(info).shape[1])
    info=info+[[0]*(INPUT_SIZE)]*need
    print('info: ' + str(leng))
    csvfile = open('show.csv','w',newline="")
    mywriter = csv.writer(csvfile,dialect='excel')
    mywriter.writerows(info)
    csvfile.close()

i="test"
ReadVoiceInfo(i+'.txt')
ReadBeatInfo(i+'_beat.txt')

for k in range(len(sum_voice_info)):
    N=len(sum_voice_info[k])
    N1=len(sum_voice_info[k][0])
    matrix_voice=array(sum_voice_info[k])
    for i in range(0,N1):
        for j in range(0,N):
            matrix_voice[j,i]= (matrix_voice[j,i] - MUSIGMA_V[i][0]) / MUSIGMA_V[i][1];  
    sum_voice_info[k]=matrix_voice.tolist()


# print("fanidfhj",len(sum_voice_info[0]))

for i in range(len(beat_info)):
    for j in range(len(beat_info[i])-1):
        start_num = len(fin_voice_info)
        fin_voice_info += sum_voice_info[i][beat_info[i][j]:beat_info[i][j+1]]

        for k in range(max_length-(beat_info[i][j+1]-beat_info[i][j])):
            voice_zero = [0 for k in range(19)]
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
