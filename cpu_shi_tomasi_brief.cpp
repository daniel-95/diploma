#include "diploma.hpp"
#include "track.hpp"

void cpu_shi_tomasi_brief(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	cv::Mat frame, grayed, descriptors;
	std::vector<cv::KeyPoint> keypoints;

	int blockSize = 4;
	double maxOverlap = 0.0;
	double minDistance = (1.0 - maxOverlap) * blockSize;
	bool useHarrisDetector = false;
	double qualityLevel = 0.6;
	double k = 0.04;
	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);		
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
	cv::Mat dst, dst_norm, dst_norm_scaled;

	std::cout << "[INFO] Shi-Tomasi + BRIEF on CPU" << std::endl;

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

		uint64_t t1 = cv::getTickCount();

		double timegap = (t1*1.0 - t0) / cv::getTickFrequency();

		if(descriptors.empty()) {
			std::cout << "[WARNING] no descriptors" << std::endl;
			continue;
		}

		if(pTracker->empty()) {
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : keypoints)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			// init the tracker with it
			pTracker->init(vfKeypoints, grayed);
			continue;
		}

		// if there are NFRAMES steps
		if(pTracker->ready()) {
			// drawing paths
			auto steps = pTracker->getSteps();
			std::cout << std::endl << "number of tracks: " << steps.size() << std::endl;

			// initialize next path search
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : keypoints)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			pTracker->init(vfKeypoints, grayed);
		} else {
			pTracker->makeStep(grayed);
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << keypoints.size();
	}
}

