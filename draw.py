import cv2
import json
import time
import pygame
import numpy as np
from matplotlib import pyplot as plt

f = open('./predict', 'r')
img = np.zeros((400,400,3), np.uint8)
lines = f.readlines()
f.close()

pygame.mixer.init()
track = pygame.mixer.music.load('./music/test.mp3')
pygame.mixer.music.play()

start = time.clock()
for i in range(int(len(lines)/3)):
	frame = lines[3*i]
	center = json.loads(lines[3*i+1])
	r = json.loads(lines[3*i+2])
	joints = []
	for j in range(21):
		for k in range(3):
			r[j][k] = r[j][k] + center[k]
			if k == 0:
				r[j][k] = r[j][k] + 200
			if k == 1:
				r[j][k] = -r[j][k] + 290
	
	cv2.line(img, (int(r[0][0]), int(r[0][1])), (int(r[1][0]), int(r[1][1])), (255,255,255), 2)	
	cv2.line(img, (int((r[0][0]+r[1][0])/2), int((r[0][1]+r[1][1])/2)), (int((r[3][0]+r[12][0])/2), int((r[3][1]+r[12][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[3][0]), int(r[3][1])), (int((r[3][0]+r[12][0])/2), int((r[3][1]+r[12][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[3][0]), int(r[3][1])), (int(r[4][0]), int(r[4][1])), (255,255,255), 2)
	cv2.line(img, (int(r[4][0]), int(r[4][1])), (int(r[5][0]), int(r[5][1])), (255,255,255), 2)
	cv2.line(img, (int(r[5][0]), int(r[5][1])), (int(r[6][0]), int(r[6][1])), (255,255,255), 2)
	cv2.line(img, (int(r[12][0]), int(r[12][1])), (int((r[3][0]+r[12][0])/2), int((r[3][1]+r[12][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[12][0]), int(r[12][1])), (int(r[13][0]), int(r[13][1])), (255,255,255), 2)
	cv2.line(img, (int(r[13][0]), int(r[13][1])), (int(r[14][0]), int(r[14][1])), (255,255,255), 2)
	cv2.line(img, (int(r[14][0]), int(r[14][1])), (int(r[15][0]), int(r[15][1])), (255,255,255), 2)
	cv2.line(img, (int(r[2][0]), int(r[2][1])), (int((r[3][0]+r[12][0])/2), int((r[3][1]+r[12][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[2][0]), int(r[2][1])), (int(r[7][0]), int(r[7][1])), (255,255,255), 2)
	cv2.line(img, (int(r[7][0]), int(r[7][1])), (int(r[8][0]), int(r[8][1])), (255,255,255), 2)
	cv2.line(img, (int(r[8][0]), int(r[8][1])), (int(r[9][0]), int(r[9][1])), (255,255,255), 2)
	cv2.line(img, (int(r[9][0]), int(r[9][1])), (int((r[10][0]+r[11][0])/2), int((r[10][1]+r[11][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[10][0]), int(r[10][1])), (int(r[11][0]), int(r[11][1])), (255,255,255), 2)
	cv2.line(img, (int(r[2][0]), int(r[2][1])), (int(r[16][0]), int(r[16][1])), (255,255,255), 2)
	cv2.line(img, (int(r[16][0]), int(r[16][1])), (int(r[17][0]), int(r[17][1])), (255,255,255), 2)
	cv2.line(img, (int(r[17][0]), int(r[17][1])), (int(r[18][0]), int(r[18][1])), (255,255,255), 2)
	cv2.line(img, (int(r[18][0]), int(r[18][1])), (int((r[19][0]+r[20][0])/2), int((r[19][1]+r[20][1])/2)), (255,255,255), 2)
	cv2.line(img, (int(r[19][0]), int(r[19][1])), (int(r[20][0]), int(r[20][1])), (255,255,255), 2)
	
	#for j in range(21):
	#	cv2.circle(img, (int(r[j][0]), int(r[j][1])), 1, (0,255,0), 2)
	#plt.imshow(img, 'brg')
	#plt.show()

	cv2.imshow('dance', img)
	img = np.zeros((400,400,3), np.uint8)
	cv2.waitKey(1)

elapsed = (time.clock() - start)
print("Time used:",elapsed)
