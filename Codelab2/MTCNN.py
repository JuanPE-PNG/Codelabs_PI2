import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from time import time

files = ["feliz.png", "triste.jpg", "mogeo.jpg"]
runs = 5

detector = MTCNN()

def inference_time(img):
    t = []
    for _ in range(runs):
        t0 = time()
        detector.detect_faces(img)
        t1 = time()
        t.append((t1 - t0)*1000)
    return sum(t)/len(t)

for f in files:
    if not os.path.exists(f):
        print(f"No se encontro {f}")
        continue

    abs_path = os.path.abspath(f)
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        size = None

    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: OpenCV could not read image: {abs_path}")
        print(f"  - exists: {os.path.exists(abs_path)}")
        print(f"  - size: {size} bytes")
        try:
            from PIL import Image
            im = Image.open(abs_path)
            im.verify()
            print(f"  - PIL: can open image (format={im.format})")
        except Exception as e:
            print(f"  - PIL: cannot open image ({e})")
        print("  → Skipping this file.\n")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Medir tiempo promedio de inferencia
    try:
        avg_t = inference_time(img_rgb)
    except Exception as e:
        print(f"Error measuring inference time for {abs_path}: {e}")
        continue

    # Detectar 1 vez para usar resultados
    try:
        res = detector.detect_faces(img_rgb)
    except Exception as e:
        print(f"Error detecting faces in {abs_path}: {e}")
        continue

    print(f"\n------ {f} ------")
    print(f"Detected (sin filtro): {len(res)} rostro(s) • tiempo promedio: {avg_t:.1f} ms")

    # Sweep de umbrales
    for thr in [0.6, 0.7, 0.8, 0.9, 0.95]:
        filtrados = [r for r in res if r.get('confidence', 0) >= thr]
        print(f"  → Con thr={thr} quedan {len(filtrados)} rostros")

    # Blur opcional por privacidad
    vis = img_rgb.copy()
    for r in res:
        x, y, w, h = r['box']
        x, y = max(0, x), max(0, y)
        face = vis[y:y+h, x:x+w]
        if face.size == 0:
            continue
        face_blur = cv2.GaussianBlur(face, (25,25), 30)
        vis[y:y+h, x:x+w] = face_blur

    # Dibujar landmarks
    for r in res:
        for name, (px,py) in r.get('keypoints', {}).items():
            cv2.circle(vis, (px,py), 3, (255, 0, 0), -1)

    plt.imshow(vis)
    plt.axis("off")
    plt.show()