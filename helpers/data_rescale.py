import cv2
import numpy as np
import os
import tqdm as tqdm

index = 0
input_folder = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_Darius_test/'

output_folder = '/home/yv312705/Code/diffusion_autoenc/datasets/Unet_darius_scaled/'

min_object_size = 4000

for filename in sorted(os.listdir(input_folder)):

    input_path = os.path.join(input_folder, filename)


    # Lade das Bild
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    object_found = False

    while not object_found:

    # Führe eine adaptive Schwellenwertbildung durch, um das Objekt hervorzuheben
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Finde die äußeren Konturen des Körpers
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_object_size]

        if len(filtered_contours) > 0:

            object_contour = max(filtered_contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(object_contour)

            object_found = True

        if not object_found:
                #image = cv2.resize(image, (int(image.shape[1] * 0.9), int(image.shape[0] * 0.9)))
                break



    # Berechne die maximale Seitenlänge des Objekts
    max_side = max(w, h)
    # Definiere die gewünschte Größe des Quadrats
    target_size = max_side + 10  # Füge den Rand von 20 Pixeln auf jeder Seite hinzu
    # Berechne die Position, um das Quadrat zentriert in das leere Bild einzufügen
    x_offset = (target_size - w) // 2
    y_offset = (target_size - h) // 2
    # Erzeuge ein leeres quadratisches Bild mit der Zielgröße
    square_image = np.zeros((target_size, target_size), dtype=np.uint8)
    # Kopiere das Objekt in das quadratische Bild
    square_image[y_offset:y_offset + h, x_offset:x_offset + w] = image[y:y + h, x:x + w]
    # Zeige das ursprüngliche und das quadratische Bild an
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, square_image)
    index += 1
   
