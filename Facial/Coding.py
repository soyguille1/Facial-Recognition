import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

patch = 'Personal'
images = []
clases = []
lista = os.listdir(patch)

comp1 = 100
for lis in lista:
    imgdb = cv2.imread(f'{patch}/{lis}')
    images.append(imgdb)
    clases.append(os.path.splitext(lis)[0])

print(clases)

def codrostros(images):
    listacod = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cod = fr.face_encodings(img)[0]
        listacod.append(cod)
    return listacod

def guardar_en_csv(nombre):
    with open('Horario.csv', 'a') as h:
        info = datetime.now()
        fecha = info.strftime('%Y-%m-%d')
        hora = info.strftime('%H:%M:%S')
        h.write(f'{nombre},{fecha},{hora}\n')
        print(f'Registrado: {nombre} en {fecha} a las {hora}')

rostroscod = codrostros(images)
cap = cv2.VideoCapture(0)

nombres_registrados = set()

while True:
    ret, frame = cap.read()
    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    facescod = fr.face_locations(rgb)
    faces = fr.face_encodings(rgb, facescod)

    for facecod, faceloc in zip(faces, facescod):
        comparacion = fr.compare_faces(rostroscod, facecod)
        simi = fr.face_distance(rostroscod, facecod)
        min = np.argmin(simi)

        if comparacion[min]:
            nombre = clases[min].upper()
            print(nombre)

            if nombre not in nombres_registrados:
                guardar_en_csv(nombre)
                nombres_registrados.add(nombre)

            yi, xf, yf, xi = faceloc
            yi, xf, yf, xi = yi * 4, xf * 4, yf * 4, xi * 4 

            indice = comparacion.index(True)

            if comp1 != indice:
                r = random.randrange(0, 255, 50)
                g = random.randrange(0, 255, 50)
                b = random.randrange(0, 255, 50)
                comp1 = indice

            if comp1 == indice:
                cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                cv2.rectangle(frame, (xi, yf - 35), (xf, yf), (r, g, b), cv2.FILLED)
                cv2.putText(frame, nombre, (xi + 6, yf - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Reconocimiento Facial", frame)
    t = cv2.waitKey(5) 
    if t == 27: 
        break

cap.release()
cv2.destroyAllWindows()
