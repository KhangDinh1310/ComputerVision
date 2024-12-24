import cv2
import numpy as np

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
def rectContour(contours):
    rectCon = []
    unRectCOn = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i) 
            else: 
                unRectCOn.append(i)
        else:
            unRectCOn.append(i)                   
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    unRectCOn.extend(rectCon[2:])
    return rectCon, unRectCOn

def getCornerPoints(cont): 
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, peri * 0.02, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) 
    myPointsNew = np.zeros((4, 1, 2), np.int32) 
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,5)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    return boxes
def showAnswers(img, myIndex, grading, ans, questions, choices):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x] == 1:  
            myColor = (0, 255, 0) 
        else: 
            myColor = (0, 0, 255)  
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2), 30, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
    return img             
def showUnMark(img, myIndex, unmark, grading, ans, questions, choices):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x, y in unmark:  
        cX = (y * secW) + secW // 2  
        cY = (x * secH) + secH // 2  
        cv2.circle(img, (cX, cY), 50, (255, 0, 0), cv2.FILLED)  

    return img
def sobel(blur): 
    sobelx_64 = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
    absx_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absx_64/absx_64.max()*255
    sobelx_8u = np.uint8(sobelx_8u1)
    
    sobely_64 = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)
    absy_64 = np.absolute(sobely_64)
    sobely_8u1 = absy_64/absy_64.max()*255
    sobely_8u = np.uint8(sobely_8u1)

    mag = np.hypot(sobelx_8u, sobely_8u)
    mag = mag/mag.max()*255
    mag = np.uint8(mag)
    
    theta = np.arctan2(sobely_64, sobelx_64)
    angle = np.rad2deg(theta)

    #cv2.imshow('Original', blur)
    #cv2.imshow('Sobel X', sobelx_8u)
    #cv2.imshow('Sobel Y', sobely_8u)
    #cv2.imshow('Gradient Magnitude', mag)
    return mag,  angle

def nonMax(mag, angle):
    M, N = mag.shape
    Non_max = np.zeros((M,N), dtype= np.uint8)
 
    for i in range(1,M-1):
        for j in range(1,N-1):
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
                b = mag[i, j+1]
                c = mag[i, j-1]
            elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
                b = mag[i+1, j+1]
                c = mag[i-1, j-1]
            elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
                b = mag[i+1, j]
                c = mag[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
                b = mag[i+1, j-1]
                c = mag[i-1, j+1]           
                
            if (mag[i,j] >= b) and (mag[i,j] >= c):
                Non_max[i,j] = mag[i,j]
            else:
                Non_max[i,j] = 0

    #cv2.imshow('Non-Max Suppression Result', Non_max)      
    return Non_max      
def hysteresis(Non_max):
    highThreshold = 50
    lowThreshold = 10
    
    M, N = Non_max.shape
    out = np.zeros((M,N), dtype= np.uint8)

    strong_i, strong_j = np.where(Non_max >= highThreshold)
    zeros_i, zeros_j = np.where(Non_max < lowThreshold)
    
    weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))
    
    out[strong_i, strong_j] = 255
    out[zeros_i, zeros_j ] = 0
    out[weak_i, weak_j] = 75

    double(out)
    #cv2.imshow("Hysteresis", out)
    return out
def double(out): 
    M, N = out.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (out[i,j] == 75):
                if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
                    out[i, j] = 255
                else:
                    out[i, j] = 0               
