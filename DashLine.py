import cv2
import numpy as np

# img_color = cv2.imread('20200904_174236_cr_0000001060.png')
# img_color = cv2.imread('20200904_174236_cr_0000001180.png')
# img_color = cv2.imread('20200904_174236_cr_0000001150.png')
# img_color = cv2.imread('20200904_174236_cr_0000001170.png')
# img_color = cv2.imread('20200904_174236_cr_0000001205.png')
# img_color = cv2.imread('20200904_174236_cr_0000001210.png')
# img_color = cv2.imread('20200904_174236_cr_0000001230.png')
# img_color = cv2.imread('20200904_174236_cr_0000001246.png')
# img_color = cv2.imread('20200904_174236_cr_0000001300.png')
# img_color = cv2.imread('20200904_174236_cr_0000001314.png')
# img_color = cv2.imread('20200904_174236_cr_0000001315.png')
# img_color = cv2.imread('20200904_174236_cr_0000001327.png')
# img_color = cv2.imread('20200904_174236_cr_0000001340.png')
img_color = cv2.imread('20200904_174236_cr_0000001350.png')
# img_color = cv2.imread('20200904_174236_cr_0000001360.png')
# img_color = cv2.imread('20200904_174236_cr_0000001370.png')
# img_color = cv2.imread('20200904_174236_cr_0000001380.png')
# img_color = cv2.imread('20200904_174236_cr_0000001390.png')
# img_color = cv2.imread('20200904_175240_cr_0000001415.png')
# img_color = cv2.imread('20200904_175240_cr_0000001460.png')
# img_color = cv2.imread('20200904_175240_cr_0000001465.png')
# img_color = cv2.imread('20200904_175240_cr_0000001470.png')

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        color = color3
    else:
        color = color1
    cv2.fillPoly(mask, vertices, color)
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def detect_stoplineB(x):
    frame = x.copy()
    img = frame.copy()
    min_dashline_length = 0 #330 #defualt 250
    #max_dashline_length = 250
    max_distance = 1000 #120 #defualt 70
    min_distance = 80

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # blur
    kernel_size = 5
    blur_frame = gray#cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # cv2.imshow("blur_frame:", blur_frame)
    
    # roi
    # vertices = np.array([[
    #     (80, frame.shape[0]),
    #     (120, frame.shape[0] - 120),
    #     (frame.shape[1] - 80, frame.shape[0] - 120),
    #     (frame.shape[1] - 120, frame.shape[0])
    # ]], dtype=np.int32)
    # vertices = np.array([[
    #     (524, frame.shape[0]*0.73), #*0.63
    #     (513, frame.shape[0]*0.51), #*0.58
    #     (553, frame.shape[0]*0.51), #*0.58
    #     (624, frame.shape[0]*0.73)  #*0.63
    # ]], dtype=np.int32)
    vertices = np.array([[
        (499, frame.shape[0]*0.73), #*0.63
        (583, frame.shape[0]*0.51), #*0.58
        (623, frame.shape[0]*0.51), #*0.58
        (599, frame.shape[0]*0.73)  #*0.63
    ]], dtype=np.int32)

    roi = region_of_interest(blur_frame, vertices)
    cv2.imshow("roi:", roi)
    # filter
    img_mask = cv2.inRange(roi, 160, 255) ## default 160, 220
    img_result = cv2.bitwise_and(roi, roi, mask=img_mask)

    cv2.imshow('bin', img_result)

    # binary
    ret, dest = cv2.threshold(img_result, 150, 255, cv2.THRESH_BINARY) ## default 160, 255
    # cv2.imshow('dest', dest)
    # canny
    low_threshold, high_threshold = 70, 210 #70, 210
    edge_img = cv2.Canny(np.uint8(dest), low_threshold, high_threshold)
    cv2.imshow('edge_img', edge_img)
    # find contours, opencv4
    contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('frame:', frame[0][0])
    # print('contours:', len(contours))
    # find contours, opencv3
    #_, contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    approx_max:any = None
    approxes = []

    if contours:
        stopline_info = [0, 0, 0, 0]
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # result = cv2.drawContours(frame, [approx], 0, (0,255,0), 4)
            # print('result:', frame)
            # cv2.imshow('result', result)
            (x, y), (w, h), theta = cv2.minAreaRect(contour)
            # print('x, y, w, h:', x, y, w, h, theta)
            if w > 30 or h > 30:
                stopline_info = [x, y, w, h]
                approx_max = approx
                print('max:', x, y, w, h, theta)
                approxes.append(approx)
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                # rect = cv2.minAreaRect(approx_max)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
        rects = [cv2.minAreaRect(appr) for appr in approxes]
        boxes = [np.int0(cv2.boxPoints(rect)) for rect in rects]
        print('boxes:', len(boxes))
        result = cv2.drawContours(frame, boxes, -1, (0, 255, 0), 3)
        # cv2.imshow('result', result)
        # print('x, y, w, h:', x, y, w, h)
        
        cx, cy = stopline_info[0] + 0.5 * stopline_info[2], stopline_info[1] + 0.5 * stopline_info[3]
        center = np.array([cx, cy])
        dashline_length = stopline_info[3]
        bot_point = np.array([frame.shape[1] // 2, frame.shape[0]])
        distance = np.sqrt(np.sum(np.square(center - bot_point)))

        # OUTPUT
        print('length : {},  distance : {}'.format(dashline_length, distance))
        # red_color = (0,0,255)
        # cv2.rectangle(img, vertices, red_color, 3)
        if dashline_length > min_dashline_length and min_distance < distance < max_distance:
        #if min_dashline_length <= dashline_length <= max_dashline_length and min_distance < distance < max_distance:
            cv2.imshow('stopline', result)
            cv2.waitKey(1)
            print('STOPLINE Detected')
            # self.stopline_detection_flag = True
            return True

    cv2.imshow('stopline', img)
    cv2.waitKey(1)
    # print('No STOPLINE.')
    return False

print('TYPE:', img_color.shape)
detect_stoplineB(img_color)
cv2.waitKey(0)
