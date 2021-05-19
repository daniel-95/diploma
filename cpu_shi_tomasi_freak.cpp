#include "diploma.hpp"
#include "track.hpp"
#include <opencv2/flann.hpp>

void cpu_shi_tomasi_freak(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	cv::Mat frame, grayed, descriptors, oldDescriptors;
	std::vector<cv::KeyPoint> keypoints, oldKeypoints;

	int blockSize = 4;
	double maxOverlap = 0.0;
	double minDistance = (1.0 - maxOverlap) * blockSize;
	bool useHarrisDetector = false;
	double qualityLevel = 0.6;
	double k = 0.04;

	bool orientationNormalized = true;
	bool scaleNormalized = true;
	float patternScale = 22.0f;
	int nOctaves = 4;
	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);		
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::FREAK::create(
		orientationNormalized, scaleNormalized, patternScale, nOctaves);
	cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
	cv::Mat dst, dst_norm, dst_norm_scaled;

	std::cout << "[INFO] Shi-Tomasi + FREAK on CPU" << std::endl;

	while(cap.read(frame)) {
		cv::cvtColor(frame, grayed, cv::COLOR_RGB2GRAY);
		dst = cv::Mat::zeros(grayed.size(), CV_32FC1);
		keypoints.clear();

		int maxCorners = grayed.rows * grayed.cols / std::max(1.0, minDistance);
		std::vector<cv::Point2f> corners;

		uint64_t t0 = cv::getTickCount();

		cv::goodFeaturesToTrack(grayed, corners, maxCorners, qualityLevel, minDistance,
			cv::Mat(), blockSize, useHarrisDetector, k);

		for (auto it = corners.begin(); it != corners.end(); ++it) {
			cv::KeyPoint newKeyPoint;
			newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
			newKeyPoint.size = blockSize;
			keypoints.push_back(newKeyPoint);
		}

		extractor->compute(grayed, keypoints, descriptors);

		if(oldKeypoints.empty()) {
			oldKeypoints = keypoints;
			keypoints.clear();
			oldDescriptors = descriptors.clone();
			continue;
		}

		std::vector<std::vector<cv::DMatch>> vKnnMatches;
		std::vector<cv::DMatch> dmMatches;
		matcher.knnMatch(oldDescriptors, descriptors, vKnnMatches, 2 );

		//-- Filter matches using the Lowe's ratio test
		const float ratio_thresh = 0.7f;
		for (size_t i = 0; i < vKnnMatches.size(); i++) {
			if (vKnnMatches[i][0].distance < ratio_thresh * vKnnMatches[i][1].distance) {
				dmMatches.push_back(vKnnMatches[i][0]);
			}
		}

		uint64_t t1 = cv::getTickCount();

		double timegap = (t1*1.0 - t0) / cv::getTickFrequency();

		if(descriptors.empty()) {
			std::cout << "[WARNING] no descriptors" << std::endl;
			continue;
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << keypoints.size() << "; features tracked: " << dmMatches.size() << std::endl;
		oldKeypoints = keypoints;
		oldDescriptors = descriptors.clone();
		keypoints.clear();

	}
}


