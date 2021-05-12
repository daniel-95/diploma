#include "track.hpp"
#include <iostream>

namespace Track {

void FeatureTracker::init(std::vector<cv::Point2f> keypoints, cv::Mat firstFrame) {
	m_clear();

	m_vSteps.push_back(keypoints);
	m_mPreviousFrame = firstFrame.clone();
}

void FeatureTracker::m_clear() {
	m_vSteps.clear();
}

const std::vector<std::vector<cv::Point2f>>& FeatureTracker::getSteps() {
	return m_vSteps;
}

bool FeatureTracker::makeStep(const cv::Mat& nextFrame) {
	// drop off states of being either empty or full
	if(m_vSteps.size() == 0 || m_vSteps.size() >= m_nFrames)
		return false;

	// calculate optical flow
	std::vector<cv::Point2f> nextStep, lastStep = m_vSteps.back();
	std::vector<uint8_t> featuresFound;

	cv::calcOpticalFlowPyrLK(m_mPreviousFrame, nextFrame, lastStep, nextStep, featuresFound, cv::noArray(), cv::Size(m_nWinSize, m_nWinSize));

	for(auto& step : m_vSteps) {
		int i = 0;
		std::cout << "[SIZE] " << step.size() << std::endl;
		for(auto it = step.begin(); it != step.end() && i < featuresFound.size(); i++) {
			if(featuresFound[i] == 0)
				it = step.erase(it);
			else
				++it;
		}
	}

	std::cout << "[INFO] features tracked: " << m_vSteps.front().size() << std::endl;

	// update theinternal state
	m_vSteps.push_back(nextStep);
	m_mPreviousFrame = nextFrame.clone();

	return true;
}

}

