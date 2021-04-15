#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>


int main() {

	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		std::cout << "Unable to open camera" << std::endl;
	}

	while (true) {
		cv::Mat frame;
		camera.read(frame);
		imshow("Camera Window", frame);

		if (cv::waitKey(1) == 'q') {
			break;
		}

	}



	return 0;
}
