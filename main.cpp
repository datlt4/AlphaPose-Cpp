#include "AlphaPose.h"

int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("../pose2.jpg");
    std::vector<bbox_t> result{
        bbox_t{16, 39, 210 - 16, 344 - 41, 0.88f, 1},
        bbox_t{266, 81, 465 - 266, 348 - 81, 0.88f, 1},
        bbox_t{543, 98, 720 - 543, 347 - 98, 0.88f, 1},
        bbox_t{81, 406, 208 - 81, 678 - 406, 0.88f, 1},
        bbox_t{264, 412, 481 - 264, 688 - 412, 0.88f, 1},
        bbox_t{513, 471, 709 - 513, 608 - 471, 0.88f, 1},
        bbox_t{50, 802, 211 - 50, 1026 - 802, 0.88f, 1},
        bbox_t{285, 810, 427 - 285, 1028 - 810, 0.88f, 1},
        bbox_t{535, 813, 694 - 535, 1026 - 813, 0.88f, 1},
    };

    std::cout << "[ ALPHAPOSE ][ TORCHSCRIPT ]" << std::endl;
    std::unique_ptr<AlphaPose> pose_est = std::make_unique<AlphaPose>("../model-zoo/fast_pose_res50/fast_res50_256x192.jit", std::vector<int>{1});

    for (int i = 0; i < 100; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = pose_est->predict(image, result);
        if (i == 0)
        {
            for (bbox_t &bb : result)
            {
                PoseEstimation::draw(image, bb);
            }
            cv::imwrite("result_torch.jpg", image);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "[ LOG ][ Pytorch ] duration: " << duration.count() << "ms" << std::endl;
    }
}
