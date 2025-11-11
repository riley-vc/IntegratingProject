import time, cv2
from pykuwahara import kuwahara

img = cv2.imread("test.jpg")
start = time.perf_counter()
out = kuwahara(img, method='mean', radius=3)
elapsed = time.perf_counter() - start
cv2.imwrite("output_py.png", out)
print(f"Python CPU time: {elapsed*1000:.2f} ms")
