#include <iostream>
#include <opencv2/opencv.hpp>

void usarCamara() {
    // Inicializa la cámara web por defecto (el índice 0 suele ser la cámara integrada de la Mac)
    cv::VideoCapture cap(0);

    // Verifica si la cámara se abrió correctamente
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo acceder a la cámara web." << std::endl;
        return;
    }

    cv::Mat frame;

    while (true) {
        // Captura el fotograma actual
        cap >> frame;

        // Comprueba que el fotograma no esté vacío
        if (frame.empty()) {
            std::cerr << "Error: Fotograma vacío." << std::endl;
            break;
        }

        // Muestra el fotograma en la ventana
        cv::imshow("Cámara en Vivo", frame);

        // Espera 30 milisegundos por una tecla; si presionas 'q' o 'ESC' (código 27), el bucle termina
        char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    // Libera la cámara y destruye todas las ventanas abiertas
    cap.release();
    cv::destroyAllWindows();
}

