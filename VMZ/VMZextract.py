import numpy as np
import cv2 
import os
#Global variables 
#File paths 


inputPath = './input/' # INPUT PATH 
outputPath = './output/' # OUTPUT PATH
#videoPath = './input/Abuse014_x264.mp4'

inputFiles = os.listdir(inputPath)
inputFiles.sort()
lenFiles = len(inputFiles)
#Video information
frame_height = 240
frame_width = 320
channels = 3

frame_count = 16

features_per_bag = 32

f = open("my_list.csv", "a")
f.write("org_video,label,start_frm,video_id\n")
Nums = []

#Function getVideoClips, getVideoFrames obtained from https://github.com/ptirupat/AnomalyDetection_CVPR18 (file name: video_util.py  )
def getVideoClips(video_path): #cuts into video segments based on frames/sec
    frames = getVideoFrames(video_path)
    clips = slidingWindow(frames, frame_count, frame_count) #arr, size=16, stride=16
    return clips, len(frames)

def getVideoFrames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            Nums.append(i)
        else:
            break
        i = i+1
    return frames
#Function slidingWindow obtained from https://github.com/ptirupat/AnomalyDetection_CVPR18 (file name: array_util.py  )
def slidingWindow(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
          
    return np.array(result)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
def writeVideoDataList(length, videoPath, vidid):
#write my_list.csv file to the input folder
    for i in range(length):
     if (i%16 == 0):
        line = videoPath+", 1,"+ str(Nums[i])+ "," +str(vidid)+"\n" 
        f.write(line)
     else:
        continue
    print("Written file no: ",vidid )

#MAIN-------
def main():
	global Nums
	for i in range(lenFiles):
		Nums = []
		name = inputFiles[i]
		path = inputPath + name
		frames, length = getVideoClips(path)
		print(length)
		writeVideoDataList(length, path, i+1)
	f.close()
	print('Data is ready!\n')
	pass

main()
