#include "diploma.hpp"
#include "track.hpp"

using namespace cv::cuda;

void cuda_orb(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	std::cout << "[INFO] ORB on CUDA" << std::endl;

	cv::Mat mFrame;
	GpuMat frame, descriptors;
	std::vector<cv::KeyPoint> keypoints;
	cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(30);

	int nFrames = 50;
	int nWinSize = 11;

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);

	while(cap.read(mFrame)) {
		cv::cvtColor(mFrame, mFrame, cv::COLOR_RGB2GRAY);
		frame.upload(mFrame);
		keypoints.clear();

		uint64_t t0 = cv::getTickCount();
		orb->detectAndCompute(frame, GpuMat(), keypoints, descriptors);
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
			pTracker->init(vfKeypoints, mFrame);
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

			pTracker->init(vfKeypoints, mFrame);
		} else {
			pTracker->makeStep(mFrame);
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << keypoints.size();
	}
}

