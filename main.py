import numpy as np
import cv2
from pylepton.Lepton3 import Lepton3
from utils.ir.utils import calulate_temp, ktoc, vis2arr, ir2arr, normalize_ir, drop_small_bboxes, overlay_bboxes , draw_box , detect_ir
import time
from detector.face_detector import ThermalFaceDetector


save_path = "img/sample"
count = 0
face_detector = ThermalFaceDetector("model/thermal_face_automl_edge_fast.tflite")
min_conf_threshold = 0.5
thresh_temp = 34
bboxes = None

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 10.0, (640,480))
with Lepton3() as l:
  while True :
    a, b = l.capture()
    print(a.shape)

    origin = a.copy()
    temp_arr = ktoc(origin)
    img = ir2arr(a, size=(160,120))
    img = normalize_ir(img)

    img_3d = img[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    # print(f"image 3d shape {img_3d.shape}")
    boxes, classes, scores, num = face_detector(img_3d)

    imH, imW, _ = img_3d.shape 
    for i in range(len(scores)):
        if ((scores[i] >= min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            bboxes = (xmin, ymin ,xmax, ymax)
    # print(f"num : {num}")
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    result = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    result = draw_box(result.copy(), boxes, scores, classes, num, min_conf_threshold)
    # print(f"image after parse model shape {img.shape}")
    # print(img)
    result = cv2.resize(result, (640,480), interpolation = cv2.INTER_NEAREST)
    result = np.array(result, dtype=np.uint8)
    if bboxes is not None:
        mean_temp, max_temp = calulate_temp(temp_arr, thresh_temp, bboxes)
        result = cv2.putText(result, f"Mean temp : {mean_temp}",
                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 0, cv2.LINE_AA)
        result = cv2.putText(result, f"Max temp : {max_temp}",
                 (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 255, 0), 0, cv2.LINE_AA)
    # out.write(result)
    cv2.imshow("frame", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# out.release()
cv2.destroyAllWindows()
