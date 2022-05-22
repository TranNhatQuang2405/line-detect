import cv2
import numpy as np

width = 480
height = 640
center_image_x = width / 2
center_image_y = height / 2
minimum_area = 250
maximum_area = 100000
lower_color = np.array([0, 0, 0])
upper_color = np.array([80, 80, 80])


def resize_frame(image, COLOUR=[255, 255, 255]):
    h, w, layers = image.shape
    if h > height:
        ratio = height/h
        image = cv2.resize(
            image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width/w
        image = cv2.resize(
            image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height/h
        wless = width/w
        if(hless < wless):
            image = cv2.resize(
                image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(
                image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(
            df), 0, 0, cv2.BORDER_CONSTANT, value=COLOUR)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(
            df), int(df), cv2.BORDER_CONSTANT, value=COLOUR)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


capture = cv2.VideoCapture("./video-1653144612.mp4")


while True:
    sucess, img = capture.read()
    try:
        img = resize_frame(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow('img', hsv)

        color_mask = cv2.inRange(hsv, lower_color, upper_color)
        cv2.imshow('img', color_mask)

        contours, hierarchy = cv2.findContours(color_mask,
                                               cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        object_area = 0
        object_x = 0
        object_y = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            found_area = w * h
            center_x = x + (w / 2)
            center_y = y + (h / 2)
            if object_area < found_area:
                object_area = found_area
                object_x = center_x
                object_y = center_y
        if object_area > 0:
            ball_location = [object_area, object_x, object_y]
        else:
            ball_location = None
        if ball_location:
            img = cv2.circle(
                img, (int(ball_location[1]), int(ball_location[2])), radius=10, color=(0, 0, 255), thickness=1)
            if (ball_location[0] > minimum_area) and (ball_location[0]
                                                      < maximum_area):
                if ball_location[1] > (center_image_x +
                                       (width/3)):
                    print("Turning right")
                elif ball_location[1] < (center_image_x +
                                         (width/3)):
                    print("Turning left")
                else:
                    print("Forward")
            elif (ball_location[0] < minimum_area):
                print("Target isn't large enough, searching")
            else:
                print("Target large enough, stopping")
        else:
            print("Target not found, searching")
        cv2.imshow("A", img)
    except:
        cv2.imshow('img', img)
        print('error')

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
