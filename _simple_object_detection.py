import cv2
import numpy as np
import os

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps 
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

videoFile = "my_video.m4v"

outputFolder = "my_output"

if not os.path.exists(outputFolder):
  os.makedirs(outputFolder)

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def runApp():
	print gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
	if cap.isOpened():
		# 取得畫面尺寸
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		# 計算畫面面積
		area = width * height

		# 初始化平均畫面
		ret, frame = cap.read()
		avg = cv2.blur(frame, (4, 4))
		avg_float = np.float32(avg)

		# 輸出圖檔用的計數器
		outputCounter = 0
		
		while(cap.isOpened()):
		  # 讀取一幅影格
		  ret, frame = cap.read()

		  # 若讀取至影片結尾，則跳出
		  if ret == False:
			break

		  # 模糊處理
		  blur = cv2.blur(frame, (4, 4))

		  # 計算目前影格與平均影像的差異值
		  diff = cv2.absdiff(avg, blur)

		  # 將圖片轉為灰階
		  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

		  # 篩選出變動程度大於門檻值的區域
		  ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

		  # 使用型態轉換函數去除雜訊
		  kernel = np.ones((5, 5), np.uint8)
		  thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
		  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

		  # 產生等高線
		  cntImg, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		  hasMotion = False
		  for c in cnts:
			# 忽略太小的區域
			if cv2.contourArea(c) < 2500:
			  continue

			hasMotion = True

			# 計算等高線的外框範圍
			(x, y, w, h) = cv2.boundingRect(c)

			# 畫出外框
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		  if hasMotion:
			# 儲存有變動的影像
			cv2.imwrite("%s/output_%04d.jpg" % (outputFolder, outputCounter), frame)
			outputCounter += 1

		  # 更新平均影像
		  cv2.accumulateWeighted(blur, avg_float, 0.01)
		  avg = cv2.convertScaleAbs(avg_float)
		  keyCode = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
               break
			cap.release()
			cv2.destroyAllWindows()
			
	  	else:
        print 'Unable to open camera'

def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_AUTOSIZE)
        # Window 
        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            ret_val, img = cap.read();
			if not ret_val:
				break
            cv2.imshow('CSI Camera',img)
			gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			gray=cv2.GaussianBlur(gray,(21,21),0)
			if firstframe is None:
				firstframe=gray
				continue
			frameDelta = cv2.absdiff(firstframe,gray)
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.dilate(thresh, None, iterations=2)
			x,y,w,h=cv2.boundingRect(thresh)
			frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.imshow("frame", frame)
			cv2.imshow("Thresh", thresh)
			cv2.imshow("frame2", frameDelta)
	    # This also acts as 
            keyCode = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
               break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print 'Unable to open camera'


if __name__ == '__main__':
    show_camera()