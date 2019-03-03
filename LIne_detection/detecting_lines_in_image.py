import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    canny=cv2.Canny(blur,50,150)
    return canny

def roi(img):
    height=img.shape[0]
    triangle=np.array([
            [(200,height),(1100,height),(550,250)]
            ])
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,triangle,255)
    masked_img=cv2.bitwise_and(img,mask)
    return masked_img

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
            
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    return np.array([left_line,right_line])


def make_coordinates(image,line_params):
    slope,intercept=line_params
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

image=cv2.imread('road.jpg')
lane_img=image.copy()
canny_img=canny(lane_img)
cropped_img=roi(canny_img)
lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
averaged_lines=average_slope_intercept(lane_img,lines)
line_img=display_lines(lane_img,averaged_lines)

final_img=cv2.addWeighted(lane_img,0.8,line_img,1,1)
while True:
    #cv2.imshow('result',roi(canny))
    #cv2.imshow('result',line_img)
    cv2.imshow('result',final_img)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()