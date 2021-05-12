#ifndef __TRACK_H__
#define __TRACK_H__

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

namespace Track {

class FeatureTracker {
private:
	int m_nFrames;
	int m_nWinSize;
	cv::Mat m_mPreviousFrame;
	std::vector<std::vector<cv::Point2f>> m_vSteps;

	void m_clear();
public:
	FeatureTracker(int frames, int winSize): m_nFrames(frames), m_nWinSize(winSize) {}
	bool ready() { return m_vSteps.size() >= m_nFrames; }
	bool empty() { return m_vSteps.empty(); }

	void init(std::vector<cv::Point2f> keypoints, cv::Mat frame);
	bool makeStep(const cv::Mat& nextFrame);
	const std::vector<std::vector<cv::Point2f>>& getSteps();
};

}

#endif
