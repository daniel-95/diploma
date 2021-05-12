#include "diploma.hpp"
#include "track.hpp"

void cpu_fast_freak(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	cv::Mat frame, grayed, descriptors;
	std::vector<cv::KeyPoint> keypoints;

	int threshold = 90;
	bool nonmaxSuppression = true;
	auto type = cv::FastFeatureDetector::TYPE_9_16;

	bool orientationNormalized = true;
	bool scaleNormalized = true;
	float patternScale = 22.0f;
	int nOctaves = 4;
	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);		
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::FREAK::create(
		orientationNormalized, scaleNormalized, patternScale, nOctaves);
	
	cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(
		threshold, nonmaxSuppression, type);

	std::cout << "[INFO] Fast + FREAK on CPU" << std::endl;

	while(cap.read(frame)) {
		cv::cvtColor(frame, grayed, cv::COLOR_RGB2GRAY);
		keypoints.clear();

		uint64_t t0 = cv::getTickCount();

		fast->detect(grayed, keypoints, cv::noArray());
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
			std::cout << "number of tracks: " << steps.size() << std::endl;

			// initialize next path search
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : keypoints)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			pTracker->init(vfKeypoints, grayed);
		} else {
			pTracker->makeStep(grayed);
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << keypoints.size() << std::endl;
	}
}


