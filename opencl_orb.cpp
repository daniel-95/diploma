#include "diploma.hpp"
#include "track.hpp"

void opencl_orb(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);	
	cv::Ptr<cv::ORB> ORBDetector = cv::ORB::create(30);
	cv::UMat frame, grayed, descriptors;
	std::vector<cv::KeyPoint> features;

	std::cout << "[INFO] ORB on OpenCL" << std::endl;
	cv::ocl::setUseOpenCL(true);

	while(cap.read(frame)) {
		features.clear();
		cv::cvtColor(frame, grayed, cv::COLOR_RGB2GRAY);

		uint64_t t0 = cv::getTickCount();
		ORBDetector->detectAndCompute(grayed, cv::noArray(), features, descriptors);
		uint64_t t1 = cv::getTickCount();

		double timegap = (t1*1.0 - t0) / cv::getTickFrequency();

		if(descriptors.empty()) {
			std::cout << "[WARNING] no descriptors" << std::endl;
			continue;
		}

		if(pTracker->empty()) {
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : features)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			// init the tracker with it
			pTracker->init(vfKeypoints, grayed.getMat(cv::ACCESS_READ));
			continue;
		}

		// if there are NFRAMES steps
		if(pTracker->ready()) {
			// drawing paths
			auto steps = pTracker->getSteps();
			std::cout << "number of tracks: " << steps.size() << std::endl;

			// initialize next path search
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : features)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			pTracker->init(vfKeypoints, grayed.getMat(cv::ACCESS_READ));
		} else {
			pTracker->makeStep(grayed.getMat(cv::ACCESS_READ));
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << features.size() << std::endl;
	}
}

