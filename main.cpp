#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>


int main() {

	// Open camera 
	cv::VideoCapture camera(0);
	if (!camera.isOpened()) {
		std::cout << "Unable to open camera!" << std::endl;
	}

	std::vector<std::string> names; // Create a vector to store every item 
	std::fstream coco_file;
	coco_file.open("coco.names", std::ios::in); // Read from file "coco.names"
	if (!coco_file) {
		std::cout << "Cannot find file" << std::endl;
	}
	else {
		std::string items;
		while (std::getline(coco_file, items)) { // Read each line until it reaches the end 
			names.push_back(items); 
		}
		coco_file.close();
	}
	
	std::string yolo_configuration = "yolov3.cfg";
	std::string yolo_weights = "yolov3.weights";
	 
	// Load the neural network 
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(yolo_configuration, yolo_weights); // Load the config and weights file
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // Use opencv as the backend 
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Use the CPU

	double scale_factor = 1 / 255.0;
	bool swap_RB = false;
	bool crop = false;
	cv::Size size = cv::Size(320, 320);
	cv::Scalar mean = cv::Scalar(0, 0, 0);

	// Create dynamic colours for different objects 
	std::vector<std::vector<int>> colour_matrix;
	for (int i = 0; i < 80; i++) {
		colour_matrix.push_back({ rand() % 256, rand() % 256, rand() % 256 }); // Generate a random number between 0-255
	}


	while (true) {
		cv::Mat frame;
		camera.read(frame);

		// Converting the frame into blob format. NOTE: The darknet network can only accept blob format.
		// Generating a 4D blob from frame
		cv::Mat blob = cv::dnn::blobFromImage(frame, scale_factor, size, mean, swap_RB, crop); // Perform some pre-processing to the image

		std::vector<std::string> output_name;
		std::vector<std::string> layer_names = net.getLayerNames(); // Getting the names of every layer in the NN
		std::vector<int> output_layer_indices = net.getUnconnectedOutLayers(); // Getting the indices of the output layer in the NN

		for (int i = 0; i < output_layer_indices.size(); i++) {
			output_name.push_back(layer_names[output_layer_indices[i] - 1]); // Store the names of the output layer inside a variable
		}

		net.setInput(blob); // Assigning blob as the input the network

		std::vector<cv::Mat> net_output;
		net.forward(net_output, output_name); // Storing the result of the 3 output layers in net_output

		// Net_output is returned in a cv matrix of (m x n) x 3 where m = no. of bounding boxes detected in the current frame 
		// And n = 85, where the first 5 columns is centre_x , centre_y, width, height of the bounding box in percentage form
		// And confidence level of the seen object and the remaining 80 columns are confidence level of all other objects.
		// This is then repeated 2 more times for the 2 other output layers

		float confidence_threshold = 0.20; // Only show objects with confidence levels of over 0.4
		std::vector<float> confidences;
		std::vector<int> class_ids;
		std::vector<cv::Rect> boxes;

		for (int i = 0; i < net_output.size(); i++) { // Loop through each of the 3 output layers
			float* data = (float*)net_output[i].data; // Cast the cv::Mat from int to float and then point to it.
			for (int row = 0; row < net_output[i].rows; row++, data += net_output[i].cols) { // Loop through each of the rows (bounding boxes)
				cv::Mat scores = net_output[i].row(row).colRange(5, net_output[i].cols); // A matrix filled with scores of all objects (5-85)
				cv::Point class_id;
				double confidence;
				cv::minMaxLoc(scores, 0, &confidence, 0, &class_id); // Returns the class id (x,y) and confidence of the maximum score of cv::Mat score
				if (confidence > confidence_threshold) {
					cv::Rect box; 
					int centre_x = (int)(data[0] * frame.cols); // X-value centre of bounding box
					int centre_y = (int)(data[1] * frame.rows); // Y-value centre of bounding box
					box.width = (int)(data[2] * frame.cols); // Width of bounding box
					box.height = (int)(data[3] * frame.rows); // Height of bounding box
					box.x = centre_x - (box.width / 2); // X-value top left of bounding box 
					box.y = centre_y - (box.height / 2); // Y-value top left of bounding box
					boxes.push_back(box); // Store each box inside boxes vector
					class_ids.push_back(class_id.x); // Store each class_id (object id (x), not bounding box id (y)) inside the class_ids vector
					confidences.push_back((float)confidence); // Store each confidence score inside the confidences vector
				}
			}
		}

		float nms_threshold = 0.25;  // The lower the threshold value, the more aggressive the suppression
		std::vector<int> indices;

		// Removes multiple bounding boxes on the same object. 
		// Returns indices which is the index of bounding boxes 
		// For e.g. there were initially 10 bounding boxes, boxes = {0,1,2,3,4,5,6,7,8,9,10}
		// Suppression got rid of 6 bounding boxes 0,2,3,7,8,10, now only 4 bounding boxes remain containing the indices, indices = {1,4,5,9}
		cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold, indices); 

		for (auto it = indices.begin(); it != indices.end(); it++) { // Loops through the indices. 
			int box_start_x = boxes[*it].x; 
			int box_start_y = boxes[*it].y; 
			int box_end_x = boxes[*it].x + boxes[*it].width;  
			int box_end_y = boxes[*it].y + boxes[*it].height;
			int colour_b = colour_matrix[class_ids[*it]][0];
			int colour_g = colour_matrix[class_ids[*it]][1];
			int colour_r = colour_matrix[class_ids[*it]][2];
			std::string object_and_score = names[class_ids[*it]] + " " + std::to_string((int)(confidences[*it] * 100)) + "%";
			float text_scale = 1.5 * (boxes[*it].width / 640.0);
			cv::rectangle(frame, cv::Point(box_start_x, box_start_y), cv::Point(box_end_x, box_end_y), cv::Scalar(colour_b, colour_g, colour_r), 2);
			cv::putText(frame, object_and_score, cv::Point(box_start_x, box_start_y - 10), cv::FONT_HERSHEY_DUPLEX, text_scale, cv::Scalar(colour_b, colour_g, colour_r), 1);
		}

		imshow("Camera Window", frame);
		if (cv::waitKey(1) == 'q') {
			break;
		}
	}

	return 0;
}


