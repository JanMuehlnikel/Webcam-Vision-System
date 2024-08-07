import cv2
import numpy as np
import os
import argparse
import logging
import time
import json

# Einrichten des Loggings
# Hier wird das Logging-Modul konfiguriert, um Debugging-Informationen auszugeben.
# Das Format umfasst Zeitstempel, Log-Level und die Nachricht selbst.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Laden der Konfiguration aus einer Datei
# Diese Funktion liest eine JSON-Konfigurationsdatei ein und gibt den Inhalt als Dictionary zurück.
# Überprüft vorher, ob die Datei existiert.
def load_config(config_path):
    if not os.path.isfile(config_path):
        logging.error(f"Konfigurationsdatei '{config_path}' nicht gefunden.")
        exit()
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Parsen der Kommandozeilenargumente
# Diese Funktion verwendet das argparse-Modul, um Kommandozeilenargumente zu parsen.
# Es ermöglicht dem Benutzer, den Pfad zur Konfigurationsdatei als Argument zu übergeben.
def parse_arguments():
    parser = argparse.ArgumentParser(description='Erweiterte Objekterkennung mit YOLOv3')
    parser.add_argument('--config', type=str, default='config.json', help='Pfad zur Konfigurationsdatei')
    args = parser.parse_args()
    return args

# Laden des YOLO-Modells
# Diese Funktion lädt die Konfigurations-, Gewichts- und Klassennamensdateien des YOLO-Modells.
# Es überprüft, ob die Dateien existieren, und gibt das Netzwerkmodell und die Klassennamen zurück.
def load_yolo_model(cfg_path, weights_path, names_path):
    if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path) or not os.path.isfile(names_path):
        logging.error("Modelldateien nicht gefunden. Bitte überprüfen Sie die Pfade.")
        exit()
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')
    logging.info("Modell erfolgreich geladen.")
    return net, classes

# Zuweisung von Farben zu jeder Klasse
# Diese Funktion weist zufällig generierte Farben zu jeder Klasse zu, um die visuellen Ergebnisse der Objekterkennung ansprechend darzustellen.
def get_class_colors(num_classes):
    np.random.seed(42)  # Seed für Reproduzierbarkeit
    return np.random.uniform(0, 255, size=(num_classes, 3))

# Objekterkennung und Zeichnen der Ergebnisse auf dem Frame
# Diese Funktion führt die Objekterkennung auf einem gegebenen Frame durch und zeichnet die erkannten Objekte zusammen mit deren Klassennamen und Konfidenzwerten.
def detect_objects(net, frame, confidence_threshold, nms_threshold, classes, class_colors):
    (H, W) = frame.shape[:2]  # Höhe und Breite des Frames ermitteln
    ln = net.getLayerNames()  # Namen aller Schichten im Netzwerk 
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Blob aus dem Bild erstellen
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    # Durch die Ausgabe der Netzwerkebenen iterieren
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # Scores für alle Klassen
            class_id = np.argmax(scores)  # ID der Klasse mit dem höchsten Score
            confidence = scores[class_id]  # Konfidenzwert der besten Klasse
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])  # Bounding-Box-Koordinaten skalieren
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS) anwenden, um überlappende Bounding-Boxes zu unterdrücken
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Zeichnen der Bounding-Boxes und Klassennamen auf dem Frame
    if len(idxs) > 0:
        idxs = idxs.flatten()
        for i in idxs:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in class_colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

# Anzeige der Bildrate (FPS) auf dem Frame
# Diese Funktion berechnet die Bildrate und zeigt sie auf dem Frame an.
def display_frame_rate(frame, start_time):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Hauptfunktion
# Laden der Konfiguration und das Modells,
# startet die Videoaufnahme und führt die Objekterkennung auf jedem Frame durch.
def main():
    args = parse_arguments()
    config = load_config(args.config)

    cfg_path = config['cfg_path']
    weights_path = config['weights_path']
    names_path = config['names_path']
    confidence_threshold = config['confidence_threshold']
    nms_threshold = config['nms_threshold']
    frame_width = config['frame_width']
    frame_height = config['frame_height']

    net, classes = load_yolo_model(cfg_path, weights_path, names_path)
    class_colors = get_class_colors(len(classes))

    cap = cv2.VideoCapture(0)  # Webcam initialisieren
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    logging.info("Starte Videoaufnahme.")
    
    while True:
        start_time = time.time()  # Startzeit für FPS-Berechnung
        ret, frame = cap.read()  # Frame von der Webcam lesen
        if not ret:
            logging.error("Fehler beim Lesen des Frames von der Webcam.")
            break

        # Objekterkennung auf dem Frame durchführen
        frame = detect_objects(net, frame, confidence_threshold, nms_threshold, classes, class_colors)
        # Bildrate auf dem Frame anzeigen
        frame = display_frame_rate(frame, start_time)
        
        cv2.imshow('Object Detection', frame)  # Frame im Fenster anzeigen
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Programm beenden, wenn 'q' gedrückt wird
            break

    cap.release()  # Webcam freigeben
    cv2.destroyAllWindows()  # Alle OpenCV-Fenster schließen
    logging.info("Videoaufnahme beendet.")

if __name__ == "__main__":
    main()
