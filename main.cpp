#include "AlphaPose.h"

int main()
{
    AlphaPose al("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/model-zoo/fast_pose_res50/fast_res50_256x192.jit");
    cv::Mat image;
    std::vector<bbox> objBoxes;
    std::vector<PoseKeypoints> poseKeypoints;
    image = cv::imread("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/samples/pose8.png");
    objBoxes.push_back(bbox(0.0f, 0.0f, (float)image.cols, (float)image.rows, 1.0f));
    objBoxes.push_back(bbox(0.0f, 0.0f, (float)image.cols, (float)image.rows, 1.0f));
    objBoxes.push_back(bbox(0.0f, 0.0f, (float)image.cols, (float)image.rows, 1.0f));
    al.predict(image, objBoxes, poseKeypoints);
    // poseKeypoints[0].to_json("poseKeypoints.json", al.cocoKeypointNames);
    cv::Mat show;
    al.draw(image, show, poseKeypoints);
    cv::imwrite("EMoi.png", show);
    cv::imshow("EMoi", show);
    int k = cv::waitKey(0);
}
