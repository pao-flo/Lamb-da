import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)

# Aquí guardaremos el último círculo (la pelota) que detectamos.
# Empieza en None porque al inicio todavía no hemos visto ninguno.
prevCircle = None

# Función para calcular la distancia entre dos puntos (x1,y1) y (x2,y2)
# usando el teorema de Pitágoras: raíz((x1-x2)² + (y1-y2)²).
# "lambda" es una forma corta de definir una función de una línea.
# Usamos int() porque los círculos vienen en tipo uint16 (solo números
# positivos). Si restas con uint16 y el resultado sería negativo,
# se "desborda" y da un número enorme. Con int() evitamos ese error.
dist = lambda x1, y1, x2, y2: ((int(x1) - int(x2))**2 + (int(y1) - int(y2))**2)**0.5

#Color Naranja en HSV
lower_orange = np.array([2, 120, 120])     # límite inferior del naranja
upper_orange = np.array([20, 255, 255])    # límite superior del naranja


# ============================================================
# FUNCIÓN PARA VERIFICAR SI UN CÍRCULO ES NARANJA
# ============================================================

def is_orange(hsv_frame, x, y, r):
    """
    Recibe la imagen en HSV, el centro del círculo (x, y) y su radio r.
    Devuelve True si suficiente parte del círculo es naranja.
    """

    # Crea una imagen negra (todo ceros) del mismo alto y ancho que el frame.
    # hsv_frame.shape[:2] toma solo (alto, ancho) y descarta los canales.
    # dtype=np.uint8 significa valores de 0 a 255 (imagen de 1 canal).
    mask_circle = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

    # Dibuja un círculo BLANCO (255) relleno sobre esa imagen negra.
    # El -1 al final significa "relleno" (en vez de solo el borde).
    # Resultado: una máscara con blanco SOLO dentro del círculo detectado.
    # int() convierte de uint16 a int normal que OpenCV acepta sin problemas.
    cv.circle(mask_circle, (int(x), int(y)), int(r), 255, -1)

    # Crea otra máscara de toda la imagen: blanco donde el píxel está dentro
    # del rango naranja, negro donde no.
    mask_orange = cv.inRange(hsv_frame, lower_orange, upper_orange)

    # Operación AND: se queda solo con lo que es blanco en AMBAS máscaras.
    # Es decir: píxeles que están dentro del círculo Y además son naranjas.
    inside = cv.bitwise_and(mask_orange, mask_circle)

    # Calcula el porcentaje de naranja dentro del círculo:
    #   countNonZero(inside)      = píxeles naranjas dentro del círculo
    #   countNonZero(mask_circle) = píxeles totales del círculo
    # max(..., 1) evita dividir entre cero si el círculo quedara vacío.
    ratio = cv.countNonZero(inside) / max(cv.countNonZero(mask_circle), 1)

    # Si más del 40% del círculo es naranja, lo consideramos naranja.
    # Devuelve True o False.
    return ratio > 0.4


# ============================================================
# CICLO PRINCIPAL (se repite por cada frame de la cámara)
# ============================================================

while True:

    # Lee un frame de la cámara.
    #   ret   = True si se pudo leer, False si falló.
    #   frame = la imagen (arreglo de NumPy en formato BGR).
    ret, frame = video.read()

    # Si no se pudo leer (cámara desconectada, sin permisos, etc.), sale del ciclo.
    if not ret:
        break

    # Convierte el frame de BGR a HSV. Se usa HSV porque separa el color (H)
    # de la iluminación (V), lo que hace mucho más fácil detectar un color.
    # OpenCV usa BGR (no RGB), por eso se usa COLOR_BGR2HSV.
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Convierte a escala de grises. HoughCircles necesita una imagen
    # de un solo canal y trabaja con bordes, no con color.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Aplica desenfoque gaussiano para reducir el ruido, evitando que
    # Hough detecte círculos falsos en texturas pequeñas.
    #   (7, 7) = tamaño del kernel, debe ser IMPAR.
    #   0      = sigma, OpenCV lo calcula solo a partir del tamaño del kernel.
    blurFrame = cv.GaussianBlur(grayFrame, (7, 7), 0)

    # Detecta círculos con la Transformada de Hough.
    # Parámetros:
    #   blurFrame       -> imagen de entrada (gris y suavizada)
    #   cv.HOUGH_GRADIENT -> método de detección (el único disponible en 2D)
    #   1.2             -> dp: resolución del acumulador. Con 1 es igual a la
    #                      de la imagen; con 1.2 es un poco menor (más rápido
    #                      y tolerante).
    #   100             -> minDist: distancia mínima en píxeles entre centros
    #                      de círculos. Evita detectar muchos círculos juntos.
    #   param1=100      -> umbral alto del detector de bordes Canny interno
    #                      (el bajo es la mitad: 50).
    #   param2=30       -> umbral del acumulador: cuántos "votos" necesita un
    #                      círculo. MÁS BAJO = más círculos (y más falsos),
    #                      MÁS ALTO = solo círculos muy claros.
    #   minRadius=30    -> radio mínimo en píxeles.
    #   maxRadius=400   -> radio máximo en píxeles.
    # Devuelve None si no encuentra nada, o un arreglo con [x, y, radio] por círculo.
    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100,
                              param1=100, param2=30, minRadius=30, maxRadius=400)

    # Solo procesamos si se detectó al menos un círculo.
    if circles is not None:

        # np.around redondea los valores decimales al entero más cercano.
        # np.uint16 los convierte a enteros sin signo de 16 bits.
        circles = np.uint16(np.around(circles))

        # "chosen" será el círculo elegido en este frame. Empieza vacío.
        chosen = None

        # circles tiene forma (1, N, 3). circles[0, :] toma los N círculos,
        # y cada "i" es [x, y, radio] de un círculo.
        for i in circles[0, :]:

            # Si este círculo NO es naranja, lo ignora y pasa al siguiente.
            # "continue" salta el resto del ciclo para este círculo.
            if not is_orange(hsvFrame, i[0], i[1], i[2]):
                continue

            # Si todavía no hay ningún círculo elegido, este (que sí es naranja)
            # se vuelve el elegido provisional.
            if chosen is None:
                chosen = i

            # Si ya tenemos un círculo del frame anterior, queremos quedarnos
            # con el naranja que esté MÁS CERCA de él. Esto da estabilidad:
            # la detección no "salta" entre círculos distintos.
            if prevCircle is not None:
                # Si el elegido actual está MÁS LEJOS del anterior que "i"...
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) > dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    # ...entonces "i" es mejor candidato y lo reemplaza.
                    chosen = i

        # Solo dibuja si al menos un círculo pasó el filtro de color naranja.
        # (Si ninguno era naranja, chosen sigue en None.)
        if chosen is not None:

            # Dibuja un punto pequeño (radio 1) en el centro del círculo.
            # Color (0, 100, 100) en BGR, grosor 3.
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)

            # Dibuja el contorno del círculo detectado.
            # chosen[2] es el radio. Color (255, 0, 255) = magenta en BGR.
            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255, 0, 255), 3)

            # Guarda este círculo para compararlo en el siguiente frame.
            prevCircle = chosen

    # Muestra el frame (con los círculos dibujados) en una ventana
    # llamada "circles".
    cv.imshow("circles", frame)

    # Espera 1 milisegundo a que se presione una tecla.
    # waitKey devuelve el código de la tecla, o -1 si no hubo ninguna.
    # "& 0xFF" se queda con los últimos 8 bits del resultado, para que la
    # comparación funcione igual en todos los sistemas.
    # ord('q') da el código numérico de la letra q.
    # Si presionas q, sale del ciclo.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================================
# LIMPIEZA FINAL
# ============================================================

# Libera la cámara para que otras aplicaciones puedan usarla.
video.release()

# Cierra todas las ventanas que abrió OpenCV.
cv.destroyAllWindows()