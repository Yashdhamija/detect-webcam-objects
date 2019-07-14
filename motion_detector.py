import time, pandas
from datetime import datetime
import cv2

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)
# print(video.isOpened())
while True:
    check, frame = video.read()
    status=0
    # converting the frame in gray for better analysis
    # print(check)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # blurring the grayscale frame to improve detection
    gray=cv2.GaussianBlur(gray,(21,21),0)


    # first frame is a referential static base frame
    if first_frame is None:
        first_frame=gray
        continue

    # get the difference from initial base frame
    delta_frame=cv2.absdiff(first_frame,gray)
    # white color the frame where diff is >= 30 thresholds and block elsewhere
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    # finding the contours (movements) in threshold frame
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        area=cv2.contourArea(contour)
        # neglecting frames with size < 100x100 pixels or > 75% the size of the frame
        if ((area < 10000) or area > gray.size*0.5):
            continue
        status=1
        # mark a rectangle in original frame for such contours
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]

    # time stamps for recorded movement shits
    if ((status_list[-1]==1 and status_list[-2]==0) or (status_list[-1]==0 and status_list[-2]==1)):
        times.append(datetime.now())

    # cv2.imshow("Gray Frame",gray)
    # cv2.imshow("Delta Frame",delta_frame)
    # cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break

# print(status_list)
# print(times)

for i in range(0,len(times),2):
    if len(times) == i+1:
        df=df.append({"Start":times[i],"End":times[i]},ignore_index=True)
    else:
        df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

# time record into a csv file
# df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
