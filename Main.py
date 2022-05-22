import math
from multiprocessing.dummy import Array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import shapely
from shapely.geometry import LineString


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):

    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def slope_lines(image, lines):

    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []  # Like /
    right_lines = []  # Like \
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                pass  # Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    # print(left_line, right_line)
    lineTMP = []
    for slope, intercept in [left_line, right_line]:

        # getting complete height of image in y1
        rows, cols = image.shape[:2]
        y1 = int(rows)  # image.shape[0]

        # taking y2 upto 60% of actual height or 60% of y1
        y2 = int(rows*0.6)  # int(0.6*y1)

        # we know that equation of line is y=mx +c so we can write it x=(y-c)/m
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        lineTMP.append([(x1, y1), (x2, y2)])
        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))
    point_between = [(lineTMP[0][0][0] + lineTMP[1][0][0]) / 2,
                     (lineTMP[0][0][1] + lineTMP[1][0][1]) / 2]

    line1 = LineString(lineTMP[0])
    line2 = LineString(lineTMP[1])
    try:
        int_pt = line1.intersection(line2)
        point_of_intersection = [int_pt.x, int_pt.y]
    except:
        point_of_intersection = [
            (lineTMP[0][1][0] + lineTMP[1][1][0]) / 2, (lineTMP[0][1][1] + lineTMP[1][1][1]) / 2]

    variance = point_between[0] - point_of_intersection[0]
    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array(
        [poly_vertices], 'int32'), color=(0, 255, 0))
    cv2.line(img, (int(point_between[0]), int(point_between[1])), (
        int(point_of_intersection[0]), int(point_of_intersection[1])), color=[255, 255, 0], thickness=10)

    return cv2.addWeighted(image, 0.7, img, 0.4, 0.), variance

    # cv2.polylines(img,np.array([poly_vertices],'int32'), True, (0,0,255), 10)
    # print(poly_vertices)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    line_img, variance = slope_lines(line_img, lines)
    return line_img, variance

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    # lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
    return lines_edges


def get_vertices(image):
    rows, cols = image.shape[: 2]
    bottom_left = [cols*0.15, rows]
    top_left = [cols*0.45, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right = [cols*0.55, rows*0.6]

    ver = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

# Lane finding Pipeline


def lane_finding_pipeline(image):

    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=200, high_threshold=400)
    cv2.imshow("Mew", canny_img)

    masked_img = region_of_interest(
        img=canny_img, vertices=get_vertices(image))
    houghed_lines, variance = hough_lines(
        img=masked_img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=180)
    output = weighted_img(
        img=houghed_lines, initial_img=image, α=0.8, β=1., γ=0.)

    return output, variance


capture = cv2.VideoCapture('./video-1653144612.mp4')

while True:
    sucess, img = capture.read()
    try:
        img, variance = lane_finding_pipeline(img)
        print(variance)
        cv2.imshow('img', img)
    except:
        cv2.imshow('img', img)
        print('error')

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
