import cv2
import numpy as np
import functions as fun

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = fun.canny(lane_image)
cropped_image = fun.region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = fun.average_slope_intercept(lane_image, lines)
line_image = fun.display_lines(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)