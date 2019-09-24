#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
 #File Name : main.py
 #Creation Date : 22-09-2019 #Created By : Rui An  
#_._._._._._._._._._._._._._._._._._._._._.
import cv2
import numpy as np
white_image = 255 * np.ones([1080, 1920, 3], dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

def extract_tooth_brush(brush_image):
    dimensions = brush_image.shape
    height = dimensions[0]
    width = dimensions[1]
    brush_image = cv2.resize(brush_image, (1920, 1080))
    result = np.zeros_like(brush_image, np.uint8)
    result = 255 * np.ones(brush_image.shape, dtype=np.uint8)
    lower = [80, 80, 40]
    upper = [90, 90, 50]
    # lower = [70, 70, 30]
    # upper = [100,100, 50]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(brush_image, lower, upper)
    output = cv2.bitwise_and(brush_image, brush_image, mask = mask)
    output_dimension = output.shape
    non_zero_tuple = np.nonzero(output)
    if (len(non_zero_tuple[0]) != 0):
        index = [non_zero_tuple[0][0], non_zero_tuple[1][0]]
        # cv2.circle(white_image, (index[1], index[0]), 30, (0, 0, 0), 1)
        cv2.putText(white_image, "I AM BRUSHING MY TEETH", (index[1], index[0]), font, 0.6, (0,0,0), 2)
        return result 
    else:
        return None
    # cv2.imwrite("circle_test.jpg", output)
    # for i in range(output_dimension[0]):
        # for j in range(output_dimension[1]):
            # if (np.any(output[i][j])):
                # cv2.circle(result, (j, i), 30, (255,255,255), 1)
                # cv2.imwrite("circle.jpg", result)
                # return 
                
def is_target(pixel): 
    b_color_diff = abs(pixel[0] - 80)
    g_color_diff = abs(pixel[1] - 80)
    r_color_diff = abs(pixel[2] - 40)
    if (b_color_diff < 10 and g_color_diff < 10 and r_color_diff < 10):
        return True
    else:
        return False


# out = cv2.VideoWriter('portraint_0_progression.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 30, (1920,1080))
cap = cv2.VideoCapture('../source/brushteeth.mp4')
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
      result = extract_tooth_brush(frame)
      # if (result is not None):
          # out.write(white_image)
  else:
    break
cv2.imwrite("portrait_1_word_big.jpg", white_image)






