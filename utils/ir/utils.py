import numpy as np
import cv2
from numpy import ma

def ktoc(val):
    # Kelvin to Celsius
    return (val - 27315) / 100.0

# Convenience functions for taking images

def vis2arr(img,size=(1024, 768), to_gray=False):
    
    arr = np.array(img)
    if size:
        arr = cv2.resize(arr, size)
    if to_gray:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr

def ir2arr(img, size=(1024, 768)):

    arr = np.array(img, dtype=np.float32)[:-2, :] # trim the 2 bottom lines
    arr = cv2.resize(ktoc(arr), size)
    
    return arr

def normalize_ir(arr):
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX) # deg. C to 0-255
    return arr.astype(np.uint8)

def draw_box(thermal_img, boxes, scores, classes, num, min_conf_threshold):
    imH, imW, _ = thermal_img.shape 
    for i in range(len(scores)):
        if ((scores[i] >= min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))
            
            cv2.rectangle(thermal_img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 0)

            # Draw label
            # object_name = "face" # Look up object name from "labels" array using class index
            # label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            # label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    return thermal_img

def detect_ir(ir_arr, thr):
    """
    Detects objects above thr temperature in ir array
    :param ir_arr: ir array in deg. C
    :param thr: threshold temperature in deg. C
    """
    
    mask = ir_arr>thr
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        polygon = cv2.approxPolyDP(c, 3, True)
        bbox = cv2.boundingRect(polygon)
        bboxes.append(bbox)
    
    return bboxes

def drop_small_bboxes(bboxes, min_size):
    """
    :param min_size: min size of bb area [px]
    """
    good = []
    for (x, y, dx, dy) in bboxes:
        if dx*dy > min_size:
            good.append((x, y, dx, dy))
    return good

def find_nth_smallest(a, n):
    return np.partition(a, n-1)[n-1]

def calulate_temp(temp_arr, thresh_temp, bboxes):
    xmin, ymin, xmax, ymax = bboxes
    temp_area = temp_arr.copy()[ymin:ymax, xmin:xmax]
    temp_area[temp_area < thresh_temp] = 0
    max_temp =  np.max(temp_area)
    # min_temp = find_nth_smallest(temp_area, 2)
    num_exceed_thresh = len(np.where(temp_area > thresh_temp)[0])
    mean_temp = np.sum(temp_area) / num_exceed_thresh
    return mean_temp, max_temp

def overlay_bboxes(arr, bboxes):
    
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    arr = arr.astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    
    COL = (255, 255, 0)
    for (x, y, dx, dy) in bboxes:
        p1 = (x,y)
        p2 = (x+dx, y+dy)
        cv2.rectangle(arr,
                      p1,
                      p2,
                      color=COL,
                      thickness=2)
    return arr

