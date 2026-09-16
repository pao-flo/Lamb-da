#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect/barcode.hpp>
#include <opencv2/imgproc.hpp>

void usarCamara();

int main() {
    // Dictionary es un objeto normal en OpenCV 5, no un Ptr.
    cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // El detector se crea UNA vez, fuera del ciclo.
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector detector(dict, detectorParams);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo acceder a la cámara web." << std::endl;
        return -1;
    }

    cv::namedWindow("Out", 0);

    cv::Mat img; // declarado antes del ciclo, se reutiliza en cada vuelta

    while (true) {
        cap >> img;

        if (img.empty()) {
            std::cerr << "Error: Fotograma vacío." << std::endl;
            break;
        }

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;

        // detectMarkers ahora es un método del objeto detector, no una función libre.
        detector.detectMarkers(img, corners, ids);

        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(img, corners, ids);
        } 
        cv::imshow("Out", img);

        // waitKey(30) en vez de waitKey(0): espera solo 30ms por una tecla,
        // así el video sigue fluyendo. Si presionas 'q' o ESC, se sale del ciclo.
        char key = (char)cv::waitKey(30);
        if (key == 27 || key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}