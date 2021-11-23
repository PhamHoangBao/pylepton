import numpy as np
import cv2
from pylepton.Lepton3 import Lepton3
from utils.ir.utils import ktoc, vis2arr, ir2arr, normalize_ir
import time
# with Lepton3() as l:
#   a,_ = l.capture()
# cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX) # extend contrast
# np.right_shift(a, 8, a) # fit data into 8 bits
# cv2.imwrite("output.jpg", np.uint8(a)) # write it!
save_path = "img/sample"
count = 0
with Lepton3() as l:
  while True :
    a, b = l.capture()
    # print(f"a: {type(a)} and b : {b / 100}")
    print(a.shape)

    # print(a[0][1])
    # cel_data = ktoc(a)
    # print(np.where(real_data > 36))
    # print((real_data-27315)/100)
    origin = a.copy()
    temp_arr = ktoc(origin)
    img = ir2arr(a, size=(160,120))
    img = normalize_ir(img)
    print(img.shape)
    # temp_img = origin[origin > 35]
    # img[np.squeeze(temp_arr) < 34] = 0
    # img[img<34] = 0
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    img = cv2.resize(img, (640,480))
    cv2.imshow("frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
