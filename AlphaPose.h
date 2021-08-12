#define _USE_MATH_DEFINES
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
#include "json.hpp"
#include "fstream"

struct PoseKeypoints
{
    PoseKeypoints() {}
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kp_scores; // proposal_score

    std::string str()
    {
        std::ostringstream os;
        for (int i = 0; i < kp_scores.size(); i++)
        {
            os << "[" << keypoints[i] << "\t" << kp_scores[i] << "]" << std::endl;
            // std::cout << os.str() << std::endl;
        }
        return os.str();
    }

    void to_json(std::string jname, std::map<int, std::string> listName)
    {
#ifdef INCLUDE_NLOHMANN_JSON_HPP_
        nlohmann::json j;
        for (int i = 0; i < kp_scores.size(); i++)
        {
            std::string i_str = listName[i];
            j[i_str] = {{"x", keypoints[i].x}, {"y", keypoints[i].y}, {"score", kp_scores[i]}};
        }
        std::string js = j.dump(2);
        std::cout << js << std::endl;

        std::ofstream f1(jname);
        f1 << std::setw(4) << j << std::endl;
        f1.close();
#endif // INCLUDE_NLOHMANN_JSON_HPP_
    }
};

struct bbox
{
    bbox(float x, float y, float w, float h, float s) : rect(x, y, w, h), score(s) {}
    bbox() {}
    std::string str()
    {
        std::ostringstream os;
        os << "x: " << rect.x << "  y: " << rect.y << "  w: " << rect.width << "  h: " << rect.height;
        return os.str();
    }
    void to_json(std::string jname)
    {
        nlohmann::json j;
        j["x"] = rect.x;
        j["y"] = rect.y;
        j["w"] = rect.width;
        j["h"] = rect.height;
        std::string js = j.dump(4);
        std::cout << js << std::endl;

        std::ofstream f1(jname);
        f1 << std::setw(4) << j << std::endl;
        f1.close();
    }
    cv::Rect_<float> rect; //x y w h
    float score;
};

class AlphaPose
{
public:
    std::map<int, std::string> cocoKeypointNames = {{0, "NOSE"}, {1, "LEFT_EYE"}, {2, "RIGHT_EYE"}, {3, "LEFT_EAR"}, {4, "RIGHT_EAR"}, {5, "LEFT_SHOULDER"}, {6, "RIGHT_SHOULDER"}, {7, "LEFT_ELBOW"}, {8, "RIGHT_ELBOW"}, {9, "LEFT_WRIST"}, {10, "RIGHT_WRIST"}, {11, "LEFT_HIP"}, {12, "RIGHT_HIP"}, {13, "LEFT_KNEE"}, {14, "RIGHT_KNEE"}, {15, "LEFT_ANKLE"}, {16, "RIGHT_ANKLE"}};
    std::map<std::string, int> cocoKeypointIndex = {{"NOSE", 0}, {"LEFT_EYE", 1}, {"RIGHT_EYE", 2}, {"LEFT_EAR", 3}, {"RIGHT_EAR", 4}, {"LEFT_SHOULDER", 5}, {"RIGHT_SHOULDER", 6}, {"LEFT_ELBOW", 7}, {"RIGHT_ELBOW", 8}, {"LEFT_WRIST", 9}, {"RIGHT_WRIST", 10}, {"LEFT_HIP", 11}, {"RIGHT_HIP", 12}, {"LEFT_KNEE", 13}, {"RIGHT_KNEE", 14}, {"LEFT_ANKLE", 15}, {"RIGHT_ANKLE", 16}};
    int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};
    AlphaPose(std::string model_path);
    void predict(cv::Mat image, std::vector<bbox> objBoxes, std::vector<PoseKeypoints> &poseKeypoints);
    void draw(const cv::Mat &matInput, cv::Mat &matOutput, const std::vector<PoseKeypoints> &poseKeypoints);

private:
    torch::jit::script::Module al;

    void preprocess(cv::Mat &img, bbox &box, torch::Tensor &imageTensor, bbox &outBox);
    void heatmap_to_coord(const torch::Tensor &hms, const bbox &box, PoseKeypoints &preds);
    void postprocess(const torch::Tensor &hm_data, const std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints);
};