#include "AlphaPose.h"

int main()
{
    AlphaPose al("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/model-zoo/fast_pose_res50/fast_res50_256x192.jit");
    cv::Mat image;
    std::vector<bbox> objBoxes;
    std::vector<PoseKeypoints> poseKeypoints;
    image = cv::imread("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/samples/pose4.jpg");
    objBoxes.push_back(bbox(65.0, 333.0, 103, 302, 0.99));
    objBoxes.push_back(bbox(82.0, 10.0, 90, 306, 0.99));
    objBoxes.push_back(bbox(295.0, 335.0, 139, 301, 0.99));
    objBoxes.push_back(bbox(323.0, 10.0, 109, 295, 0.97));
    al.predict(image, objBoxes, poseKeypoints);
    poseKeypoints[0].to_json("poseKeypoints.json", al.cocoKeypointNames);
    cv::Mat show;
    al.draw(image, show, poseKeypoints);
    cv::imwrite("EMoi.png", show);
    cv::imshow("EMoi", show);
    int k = cv::waitKey(0);
}
