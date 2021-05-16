#include "diploma.hpp"
#include "track.hpp"

using namespace cv::cuda;

void cuda_surf(char *fileName) {
	cv::VideoCapture cap(fileName);
	if(!cap.isOpened()) {
		std::cout << "[ERROR] bad video file" << std::endl;
		return;
	}

	std::cout << "[INFO] SURF on CUDA" << std::endl;

	int nFrames = 50;
	int nWinSize = 11;

	cv::Mat mFrame;
	GpuMat frame, features, descriptors;
	SURF_CUDA surf(	/*_hessianThreshold,*/ 11000,
		/*_nOctaves =*/ 4,
		/*_nOctaveLayers =*/ 2,
		/*_extended =*/ false,
		/*_keypointsRatio =*/ 0.01f,
		/*_upright =*/ false 
	);

	auto pTracker = std::make_unique<Track::FeatureTracker>(nFrames, nWinSize);

	while(cap.read(mFrame)) {
		cv::cvtColor(mFrame, mFrame, cv::COLOR_RGB2GRAY);
		frame.upload(mFrame);

		uint64_t t0 = cv::getTickCount();
		surf(frame, GpuMat(), features, descriptors);
		uint64_t t1 = cv::getTickCount();

		double timegap = (t1*1.0 - t0) / cv::getTickFrequency();

		if(descriptors.empty()) {
			std::cout << "[WARNING] no descriptors" << std::endl;
			continue;
		}

		if(pTracker->empty()) {
			std::vector<cv::KeyPoint> keypoints;
			surf.downloadKeypoints(features, keypoints);
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
			std::vector<cv::KeyPoint> keypoints;
			surf.downloadKeypoints(features, keypoints);
			std::vector<cv::Point2f> vfKeypoints;

			for(auto p : keypoints)
				vfKeypoints.emplace_back(p.pt.x, p.pt.y);

			pTracker->init(vfKeypoints, mFrame);
		} else {
			pTracker->makeStep(mFrame);
		}

		std::cout << "elapsed time: " << timegap << "; featured found: " << features.cols;
	}
}

