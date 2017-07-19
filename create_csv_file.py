import json
import csv
from numpy import *

action_info = []
voice_info = []
info = []
test_voice_info = []
def ReadActionInfo(file_name): 
    file = open('res/'+file_name)
    lines = file.readlines()
    action_info_index = len(action_info)-1
    # print(len(lines))
    for i in range(len(lines)):
        if i % 3 == 0:
            # 确定帧数
            # print("hfaidhfia")
            action_info_index += 1
            # if action_info_index==10:
            #     break
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
    # print(len(lines))
    for line in lines:
        line = line.replace('\n','')
        s = json.loads(line)
        voice_info.append(s)
    file.close()

def WriteCsv():
    # print(len(action_info))
    # print(len(voice_info))
    for i in range(len(action_info)):
        # print(i)
        info.append(voice_info[i]+action_info[i])
    # print(info[0])
    csvfile = open('train.csv','w',newline='')
    mywriter = csv.writer(csvfile,dialect='excel')
    mywriter.writerows(info)
    csvfile.close()

musigma=[[-517.24007460445262, 76.48976658679949],
[93.308331646297461, 41.097895203462457],
[5.5866864669164551, 30.527732762988489],
[14.270753532153032, 22.692826268466149],
[2.5808775701858147, 19.419580552530491],
[2.6395293366275356, 19.064679726744743],
[5.2394638384629459, 16.919467530406116],
[2.3986354189061645, 16.029394006802814],
[1.8094776173432667, 13.729831638077082],
[6.5419727009561237, 13.719833962861056],
[-0.0023932105976085727, 12.704075514809471],
[1.835649509944129, 11.822289882035626],
[2.8769747805621799, 11.396369715781358],
[0.082515104004446976, 0.055994849106685646],
[0.24563899682756782, 0.19105430890401423],
[180.17657493884909, 344.73627901743424],
[257.91006180059873, 253.49158643034423],
[130.1750768427041, 28.024028168713034],
[0.90267787929125132, 0.097445427571169668]]
def ReadTestVoiceInfo(file_name):
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n','')
        s = json.loads(line)
        for i in range(0,19):
            s[i]=s[i]*musigma[i][1]+musigma[i][0]
        test_voice_info.append(s)
    file.close()
    csvfile = open('test.csv','w',newline='')
    mywriter = csv.writer(csvfile,dialect='excel')
    mywriter.writerows(test_voice_info)
    csvfile.close()

for i in range(13):
    # print('i'+str(i+1))
    ReadActionInfo(str(i+1))
    ReadVoiceInfo(str(i+1)+'.txt')
N=len(voice_info)
N1=len(voice_info[0])
matrix_voice=array(voice_info)
for i in range(0,19):
    mu=average(matrix_voice[:,i])
    sigma=std(matrix_voice[:,i])
    print ([mu,sigma])
    for j in range(0,N):
        matrix_voice[j,i]= (matrix_voice[j,i] - mu) / sigma;  
voice_info=matrix_voice.tolist()
M=len(action_info)
M1=len(action_info[0])
matrix_action=array(action_info)
for i in range(0,66):
    mu=average(matrix_action[:,i])
    sigma=std(matrix_action[:,i])
    # print ([mu,sigma])
    for j in range(0,M):
        matrix_action[j,i]= (matrix_action[j,i] - mu) / sigma;  
action_info=matrix_action.tolist()
WriteCsv()
# ReadTestVoiceInfo('data1/0.txt')
