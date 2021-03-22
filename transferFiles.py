"""
Use this file to move the dataset videos into their respective train and test split folders.
"""
import shutil 
import os 
import sys
from os import listdir
fromm = './Dataset/' #the sub directory where all the dataset 
to = './Dataset/TrainingFeatures/AnomalyFeatures/'
f = open('./Anomaly_Detection_splits/Anomaly_Train.txt', 'r')
list = f.read().split('\n')
f.close()
list = list[:-1] # remove ending empty line 

#this file is used to move the features extracted in pickle form into their
#respective sub folders
for i in range(len(list)):
	name = list[i].rsplit('/')
	folder = name[0]+'Features/'
	folder = fromm+folder
	name = name[1]
	name = name[:-4]
	name = name + '.txt'
	shutil.move(folder+name, to+name)
	print("Moved: ", name)

#The code below is used to compare the two extracted features sub directories and make sure
#For all the videos
"""
AllTest_Video_Path = './Dataset/TrainingFeatures/AnomalyFeatures/'
path = './Dataset/'
All_Test_files= listdir(AllTest_Video_Path)
All_Test_files.sort()
l = len(list)
print(len(All_Test_files))
for i in range(l):
	name = list[i].rsplit('/')
	name = name[1]
	name = name[:-4]
	name = name + '.txt'
	real = to+name
	filee = to+ All_Test_files[i]
	if real != filee:
		print('Missing File is: \n')
		print(real)
"""

