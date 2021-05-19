#include "diploma.hpp"
#include "track.hpp"
#include <opencv2/flann.hpp>

using namespace cv::cuda;

void cpu_surf(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);
	cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> SURFDetector = cv::xfeatures2d::SurfFeatureDetector::create(11000);
	cv::Mat frame, frameClone, grayed, descriptors, oldDescriptors;
	std::vector<cv::KeyPoint> keypoints, oldKeypoints;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

	std::cout << "[INFO] SURF on CPU" << std::endl;

	while(cap.read(frame)) {
		keypoints.clear();
		cv::cvtColor(frame, grayed, cv::COLOR_RGB2GRAY);

		uint64_t t0 = cv::getTickCount();
		SURFDetector->detectAndCompute(grayed, cv::noArray(), keypoints, descriptors);
		uint64_t t1 = cv::getTickCount();

		double timegap = (t1*1.0 - t0) / cv::getTickFrequency();

		if(descriptors.empty()) {
			std::cout << "[WARNING] no descriptors" << std::endl;
			continue;
		}

		if(oldKeypoints.empty()) {
			oldKeypoints = keypoints;
			keypoints.clear();
			oldDescriptors = descriptors.clone();
			continue;
		}

		std::vector<std::vector<cv::DMatch>> vKnnMatches;
		std::vector<cv::DMatch> dmMatches;
		matcher->knnMatch(oldDescriptors, descriptors, vKnnMatches, 2 );

		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.7f;
		for (size_t i = 0; i < vKnnMatches.size(); i++) {
			if (vKnnMatches[i][0].distance < ratio_thresh * vKnnMatches[i][1].distance) {
				dmMatches.push_back(vKnnMatches[i][0]);
			}
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << keypoints.size() << "; features tracked: " << dmMatches.size() << std::endl;
		oldKeypoints = keypoints;
		oldDescriptors = descriptors.clone();
		keypoints.clear();
	}
}


