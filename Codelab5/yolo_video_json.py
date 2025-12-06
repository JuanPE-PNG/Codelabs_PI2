import time
import json
import argparse
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def record_and_detect(duration=10, src=0, out_json='resultados_video.json', out_plot='person_counts.png'):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 20.0

    detecciones_frames = []
    person_counts = []

    start = time.time()
    frame_idx = 0
    print(f"Grabando y detectando durante {duration} segundos (FPS estimada: {fps})...")

    while True:
        now = time.time()
        if now - start > duration:
            break

        ret, frame = cap.read()
        if not ret:
            print("Fin de la captura o no se recibió frame.")
            break

        results = model(frame)

        frame_detections = []
        persons_in_frame = 0
        for r in results[0].boxes:
            cls_idx = int(r.cls)
            cls_name = model.names[cls_idx]
            score = float(r.conf)
            bbox = [float(x) for x in r.xyxy.tolist()[0]] if hasattr(r, 'xyxy') else r.xyxy.tolist()

            obj = {
                "frame": frame_idx,
                "time": round(now - start, 3),
                "clase": cls_name,
                "score": score,
                "bbox": bbox
            }
            frame_detections.append(obj)

            if cls_name.lower() == 'person' or cls_name.lower() == 'persona':
                persons_in_frame += 1

        detecciones_frames.append({"frame_index": frame_idx, "time": round(now - start, 3), "detections": frame_detections})
        person_counts.append(persons_in_frame)

        frame_idx += 1

    cap.release()

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(detecciones_frames, f, ensure_ascii=False, indent=4)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(person_counts)), person_counts, marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Número de personas')
    plt.title('Personas detectadas por frame')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    print(f"Detecciones guardadas en: {out_json}")
    print(f"Gráfico guardado en: {out_plot}")


def main():
    parser = argparse.ArgumentParser(description='Grabar 10s de video y detectar con YOLO, guardar JSON y gráfico')
    parser.add_argument('--duration', '-d', type=int, default=10, help='Duración en segundos (por defecto 10)')
    parser.add_argument('--src', type=int, default=0, help='Índice de la cámara o ruta de video (por defecto 0)')
    parser.add_argument('--out-json', type=str, default='resultados_video.json', help='Archivo JSON de salida')
    parser.add_argument('--out-plot', type=str, default='person_counts.png', help='Archivo de imagen del gráfico')

    args = parser.parse_args()

    record_and_detect(duration=args.duration, src=args.src, out_json=args.out_json, out_plot=args.out_plot)


if __name__ == '__main__':
    main()
