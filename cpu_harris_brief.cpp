#include "diploma.hpp"
#include "track.hpp"

void cpu_harris_brief(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	cv::Mat frame, grayed, descriptors;
	std::vector<cv::KeyPoint> keypoints;

	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	int thresh = 180;
	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);		
	cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
	cv::Mat dst, dst_norm, dst_norm_scaled;

	std::cout << "[INFO] Harris + BRIEF on CPU" << std::endl;

	while(cap.read(frame)) {
		cv::cvtColor(frame, grayed, cv::COLOR_RGB2GRAY);
		dst = cv::Mat::zeros(grayed.size(), CV_32FC1);
		keypoints.clear();

		uint64_t t0 = cv::getTickCount();

		cv::cornerHarris(grayed, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
		cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
		cv::convertScaleAbs(dst_norm, dst_norm_scaled);

		for (size_t j = 0; j < dst_norm_scaled.rows; j++) {
			for (size_t i = 0; i < dst_norm_scaled.cols; i++) {
				int response = (int)dst_norm.at<float>(j, i);

				if (response > thresh) {
					cv::KeyPoint newKeyPoint;
					newKeyPoint.pt = cv::Point2f(i, j);
					newKeyPoint.size = 2 * apertureSize;
					newKeyPoint.response = response;

					// non-maximum suppression
					bool bOverlap = false;
					for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
						double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
						// if there is any overlap
						if (kptOverlap > 0) {
							bOverlap = true;

							if (newKeyPoint.response > (*it).response) {
								*it = newKeyPoint;
								break;
							}
						}

					}

					if (!bOverlap)
						keypoints.push_back(newKeyPoint);
				}
			}
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


