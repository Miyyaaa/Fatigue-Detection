"""
GÉNÉRATEUR DE DATASET PERSONNALISÉ POUR LA DÉTECTION DE FATIGUE
================================================================

Ce script capture des images de votre visage via webcam pour créer
un dataset d'entraînement personnalisé.

Instructions:
1. Lancez le script
2. Gardez un visage alerte (yeux ouverts) et appuyez sur 'A' pour capturer
3. Simulez la fatigue (yeux mi-clos, bâillements) et appuyez sur 'F' pour capturer
4. Appuyez sur 'Q' pour terminer

Usage:
    python generate_dataset.py --output ./data --samples 200
"""

import cv2
import os
import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./data')
    parser.add_argument('--samples', type=int, default=200, help='Nombre cible par classe')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    
    # Créer les dossiers
    output = Path(args.output)
    for split in ['train', 'val']:
        for cls in ['alert', 'fatigued']:
            (output / split / cls).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(args.camera)
    
    # Compteurs
    alert_count = 0
    fatigue_count = 0
    target = args.samples
    
    # Initialiser MediaPipe pour extraire le visage
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        import urllib.request
        import mediapipe as mp_img
        
        model_path = '/tmp/face_landmarker.task'
        if not os.path.exists(model_path):
            print("Téléchargement du modèle MediaPipe...")
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
        
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        print("MediaPipe initialisé")
    except Exception as e:
        print(f"Erreur MediaPipe: {e}")
        landmarker = None
    
    print("\n" + "="*60)
    print("  GÉNÉRATION DE DATASET - DÉTECTION DE FATIGUE")
    print("="*60)
    print("\nContrôles:")
    print("  A - Capturer visage ALERTE (yeux ouverts)")
    print("  F - Capturer visage FATIGUÉ (yeux fermés/bâillement)")
    print("  Q - Quitter")
    print(f"\nObjectif: {target} images par classe")
    print("="*60 + "\n")
    
    last_capture = 0
    capture_delay = 0.15  # 150ms entre captures
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # Extraire le visage
        face_roi = None
        if landmarker:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_img.Image(image_format=mp_img.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_image)
            
            if results.face_landmarks:
                h, w = frame.shape[:2]
                lm = results.face_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                xs, ys = zip(*pts)
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                pad = int((x2-x1)*0.15)
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                face_roi = frame[y1:y2, x1:x2]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # HUD
        cv2.rectangle(display, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.putText(display, f"Alert: {alert_count}/{target}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Fatigue: {fatigue_count}/{target}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display, "[A]lert [F]atigue [Q]uit", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Dataset Generator", display)
        
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord('q'):
            break
        
        if face_roi is not None and current_time - last_capture > capture_delay:
            if key == ord('a') and alert_count < target:
                # 80% train, 20% val
                split = 'train' if alert_count < target * 0.8 else 'val'
                path = output / split / 'alert' / f'{alert_count:04d}.jpg'
                face_resized = cv2.resize(face_roi, (224, 224))
                cv2.imwrite(str(path), face_resized)
                alert_count += 1
                last_capture = current_time
                print(f"Alert: {alert_count}/{target}")
            
            elif key == ord('f') and fatigue_count < target:
                split = 'train' if fatigue_count < target * 0.8 else 'val'
                path = output / split / 'fatigued' / f'{fatigue_count:04d}.jpg'
                face_resized = cv2.resize(face_roi, (224, 224))
                cv2.imwrite(str(path), face_resized)
                fatigue_count += 1
                last_capture = current_time
                print(f"Fatigue: {fatigue_count}/{target}")
    
    cap.release()
    cv2.destroyAllWindows()
    if landmarker:
        landmarker.close()
    
    print("\n" + "="*60)
    print(f"Dataset créé: {alert_count} alert, {fatigue_count} fatigued")
    print(f"Pour entraîner: python train.py --data_dir {args.output} --epochs 20")
    print("="*60)


if __name__ == "__main__":
    main()
