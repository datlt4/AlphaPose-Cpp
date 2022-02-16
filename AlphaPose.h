#ifndef ALPHAPOSE_H
#define ALPHAPOSE_H
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <assert.h>
#include <map>
#include <chrono>
#include <atomic>
#include "fstream"
#include "common.h"
struct PoseKeypoints
{
    PoseKeypoints() {}
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kp_scores; // proposal_score
};

struct bbox
{
    bbox(float x, float y, float w, float h, float s) : rect(x, y, w, h), score(s) {}
    bbox() {}
    cv::Rect_<float> rect; // x y w h
    float score;
};

namespace PoseEstimation
{
    constexpr std::array<int, 32> skeleton{0, 1,
                                           0, 2,
                                           1, 3,
                                           2, 4,
                                           5, 6,
                                           5, 7,
                                           6, 8,
                                           7, 9,
                                           8, 10,
                                           11, 12,
                                           5, 11,
                                           6, 12,
                                           11, 13,
                                           13, 15,
                                           12, 14,
                                           14, 16};

    inline std::array<cv::Scalar, 16> skeleton_color{
        cv::Scalar(148, 0, 59),
        cv::Scalar(137, 0, 157),
        cv::Scalar(154, 0, 110),
        cv::Scalar(97, 0, 151),
        cv::Scalar(22, 0, 163),
        cv::Scalar(32, 101, 153),
        cv::Scalar(43, 153, 100),
        cv::Scalar(46, 148, 156),
        cv::Scalar(28, 157, 31),
        cv::Scalar(22, 54, 163),
        cv::Scalar(30, 150, 0),
        cv::Scalar(141, 153, 0),
        cv::Scalar(61, 151, 0),
        cv::Scalar(104, 152, 0),
        cv::Scalar(157, 102, 0),
        cv::Scalar(153, 56, 0),
    };

    inline void draw(cv::Mat &drawMat, const std::vector<cv::Point2f> &keypoints)
    {
        for (int i = 0; i < PoseEstimation::skeleton.size(); i += 2)
        {
            cv::line(drawMat,
                     cv::Point2i((int)keypoints[PoseEstimation::skeleton[i]].x, (int)keypoints[PoseEstimation::skeleton[i]].y),
                     cv::Point2i((int)keypoints[PoseEstimation::skeleton[i + 1]].x, (int)keypoints[PoseEstimation::skeleton[i + 1]].y),
                     PoseEstimation::skeleton_color[i / 2], 2);
        }
        for (int i = 0; i < keypoints.size(); i++)
        {
            cv::circle(drawMat, cv::Point2i((int)keypoints[i].x, (int)keypoints[i].y), 5, cv::Scalar(255, 0, 255), cv::FILLED, 8);
        }
    }

    inline void draw(cv::Mat &drawMat, const bbox_t &bbox)
    {
        PoseEstimation::draw(drawMat, bbox.keypoints);
    }
}

class AlphaPose
{
public:
    std::map<int, std::string> cocoKeypointNames = {{0, "NOSE"}, {1, "LEFT_EYE"}, {2, "RIGHT_EYE"}, {3, "LEFT_EAR"}, {4, "RIGHT_EAR"}, {5, "LEFT_SHOULDER"}, {6, "RIGHT_SHOULDER"}, {7, "LEFT_ELBOW"}, {8, "RIGHT_ELBOW"}, {9, "LEFT_WRIST"}, {10, "RIGHT_WRIST"}, {11, "LEFT_HIP"}, {12, "RIGHT_HIP"}, {13, "LEFT_KNEE"}, {14, "RIGHT_KNEE"}, {15, "LEFT_ANKLE"}, {16, "RIGHT_ANKLE"}};
    std::map<std::string, int> cocoKeypointIndex = {{"NOSE", 0}, {"LEFT_EYE", 1}, {"RIGHT_EYE", 2}, {"LEFT_EAR", 3}, {"RIGHT_EAR", 4}, {"LEFT_SHOULDER", 5}, {"RIGHT_SHOULDER", 6}, {"LEFT_ELBOW", 7}, {"RIGHT_ELBOW", 8}, {"LEFT_WRIST", 9}, {"RIGHT_WRIST", 10}, {"LEFT_HIP", 11}, {"RIGHT_HIP", 12}, {"LEFT_KNEE", 13}, {"RIGHT_KNEE", 14}, {"LEFT_ANKLE", 15}, {"RIGHT_ANKLE", 16}};
    AlphaPose(std::string model_path, std::vector<int> person_index_class);
    void predict(cv::Mat &image, std::vector<bbox> &objBoxes, std::vector<PoseKeypoints> &poseKeypoints);
    // std::vector<bbox_t> predict(VizgardFrame& imageVizgard, std::vector<bbox_t>& bboxes);
    std::vector<bbox_t> predict(cv::Mat &imageVizgard, std::vector<bbox_t> &bboxes);

private:
    std::vector<int> person_index_class;
    torch::jit::script::Module al;
    void preprocess(cv::Mat &img, bbox &box, torch::Tensor &imageTensor, bbox &outBox);
    void heatmap_to_coord(const torch::Tensor &hms, const bbox &box, PoseKeypoints &preds);
    void postprocess(const torch::Tensor &hm_data, const std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints);
};

#endif // ALPHAPOSE_H
