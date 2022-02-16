#ifndef ALPHAPOSE_TRT_H
#define ALPHAPOSE_TRT_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <assert.h>
#include <map>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <map>
#include <numeric>
#include <iomanip>
#include "dirent.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "fstream"
#include "common.h"
#include "Logger.h"
#define THRESHOLD_RATIO 0.1

extern alphaposeTrtLoger::Logger *allogger;

using Severity = nvinfer1::ILogger::Severity;

struct TRTDestroy
{
    template <class T>
    void operator()(T *obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

template <typename T>
TRTUniquePtr<T> makeUnique(T *t)
{
    return TRTUniquePtr<T>{t};
}

struct Parser
{
    // TrtUniquePtr<nvcaffeparser1::ICaffeParser> caffeParser;
    // TrtUniquePtr<nvuffparser::IUffParser> uffParser;
    TRTUniquePtr<nvonnxparser::IParser> onnxParser;
    operator bool() const
    {
        // return caffeParser || uffParser || onnxParser;
        return !!(onnxParser);
    }
};

class TrtLogger : public nvinfer1::ILogger
{
    // trtlogger::Logger *logger;
public:
    // TRTLogger(trtlogger::Logger* logger){
    //     this->logger = logger;
    // }
    // https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/common/logging.h#L258
    void log(Severity severity, const char *msg) noexcept override
    {
        static alphaposeTrtLoger::LogLevel map[] = {
            alphaposeTrtLoger::FATAL, alphaposeTrtLoger::ERROR, alphaposeTrtLoger::WARNING, alphaposeTrtLoger::INFO, alphaposeTrtLoger::TRACE};
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
        {
            alphaposeTrtLoger::LogTransaction(allogger, map[(int)severity], __FILE__, __LINE__, __FUNCTION__).GetStream() << msg;
        }
    }
    nvinfer1::ILogger &getTRTLogger()
    {
        return *this;
    }
};

namespace PoseEstimation
{
    const std::map<int, std::string> keypoint_name{
        std::make_pair(0, "nose"),
        std::make_pair(1, "left_eye"),
        std::make_pair(2, "right_eye"),
        std::make_pair(3, "left_ear"),
        std::make_pair(4, "right_ear"),
        std::make_pair(5, "left_shoulder"),
        std::make_pair(6, "right_shoulder"),
        std::make_pair(7, "left_elbow"),
        std::make_pair(8, "right_elbow"),
        std::make_pair(9, "left_wrist"),
        std::make_pair(10, "right_wrist"),
        std::make_pair(11, "left_hip"),
        std::make_pair(12, "right_hip"),
        std::make_pair(13, "left_knee"),
        std::make_pair(14, "right_knee"),
        std::make_pair(15, "left_ankle"),
        std::make_pair(16, "right_ankle"),
    };

    constexpr std::array<int, 32> skeleton{
        0, 1,
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

    const std::array<cv::Scalar, 16> skeleton_color{
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

    struct PoseKeypoints
    {
        PoseKeypoints() {}
        std::vector<cv::Point2f> keypoints;
        std::vector<float> kp_scores; // proposal_score
        friend std::ostream &operator<<(std::ostream &os, const PoseKeypoints &kpts)
        {
            for (int i = 0; i < 17; ++i)
            {
                os << std::setw(15) << std::left << PoseEstimation::keypoint_name.at(i) << ": "
                   << std::setw(8) << std::right << std::setprecision(5) << kpts.keypoints.at(i).x
                   << std::setw(8) << std::right << std::setprecision(5) << kpts.keypoints.at(i).y
                   << std::setw(8) << std::right << std::setprecision(5) << kpts.kp_scores.at(i) << std::endl;
            }
            return os;
        }
    };

    struct bbox
    {
        bbox(float x, float y, float w, float h, float s) : rect(x, y, w, h), score(s) {}
        bbox() {}
        cv::Rect_<float> rect; // x y w h
        float score;
    };

    inline void draw(cv::Mat &drawMat, const std::vector<cv::Point2f> &keypoints, const std::vector<float> &kp_scores)
    {
        const float *max_pos = std::max_element(kp_scores.data(), kp_scores.data() + kp_scores.size());
        float max_score = kp_scores[max_pos - kp_scores.data()];
        for (int i = 0; i < PoseEstimation::skeleton.size(); i += 2)
        {
            if (kp_scores[PoseEstimation::skeleton[i]] < (THRESHOLD_RATIO * max_score) || kp_scores[PoseEstimation::skeleton[i + 1]] < (THRESHOLD_RATIO * max_score))
                continue;
            cv::line(drawMat,
                     cv::Point2i((int)keypoints[PoseEstimation::skeleton[i]].x, (int)keypoints[PoseEstimation::skeleton[i]].y),
                     cv::Point2i((int)keypoints[PoseEstimation::skeleton[i + 1]].x, (int)keypoints[PoseEstimation::skeleton[i + 1]].y),
                     PoseEstimation::skeleton_color[i / 2], 2);
        }
        for (int i = 0; i < keypoints.size(); i++)
        {
            if (kp_scores[i] < (THRESHOLD_RATIO * max_score))
                continue;
            cv::circle(drawMat, cv::Point2i((int)keypoints[i].x, (int)keypoints[i].y), 5, cv::Scalar(255, 0, 255), cv::FILLED, 8);
        }
    }

    inline void draw(cv::Mat &drawMat, const bbox_t &bbox)
    {
        PoseEstimation::draw(drawMat, bbox.keypoints, bbox.kp_scores);
    }

    inline int64_t volume(const nvinfer1::Dims &d);

    inline std::string log_cuda_bf(nvinfer1::Dims const &dim_shape, void *cuda_buffer, int number_p)
    {
        std::ostringstream oss;
        if (!cuda_buffer)
            oss << "Null cuda buffer !" << std::endl;
        oss << "Buffer size: ";
        for (size_t i = 0; i < dim_shape.nbDims - 1; ++i)
            oss << dim_shape.d[i] << "x";
        oss << dim_shape.d[dim_shape.nbDims - 1] << ".  Some elements: ";
        int64_t v = volume(dim_shape);
        std::vector<float> cpu_output(v > 0 ? v : -v);
        cudaMemcpy(cpu_output.data(), (float *)cuda_buffer, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < number_p; i++)
            oss << cpu_output[i] << " ";
        oss << std::endl;
        return oss.str();
    }
}

class AlphaPoseTRT
{
protected:
    Parser parser;
    TRTUniquePtr<nvinfer1::INetworkDefinition> prediction_network;
    TRTUniquePtr<nvinfer1::ICudaEngine> prediction_engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> prediction_context{nullptr};

    TrtLogger gLogger = TrtLogger();
    int batch_size = 1;
    std::vector<nvinfer1::Dims> prediction_input_dims;
    std::vector<nvinfer1::Dims> prediction_output_dims;

    std::vector<void *> input_buffers; // buffers for input and output data
    std::vector<void *> output_buffers;

    cudaStream_t stream;

public:
    AlphaPoseTRT(std::vector<int> person_index_class) : person_index_class{person_index_class}
    {
        cudaStreamCreate(&stream);
    };

    ~AlphaPoseTRT()
    {
        LOG(INFO) << "~AlphaPoseTRT()";
        cudaStreamDestroy(stream);
        LOG(INFO) << "~AlphaPoseTRT()";
    }

    /*virtual*/ bool parseOnnxModel(const std::string &model_path, const int minBatchSize, const int optBatchSize, const int maxBatchSize, const int workspace);
    /*virtual*/ bool saveEngine(const std::string &fileName);
    /*virtual*/ bool loadEngine(const std::string &fileName);

    std::vector<float> prepareImage(cv::Mat &img, PoseEstimation::bbox &box, PoseEstimation::bbox &outBox);
    bool processInput(float *hostDataBuffer, const int batchSize, cudaStream_t &stream);
    bool processOutput(std::vector<void *> gpu_output, cv::Mat cv_mat);

    std::vector<bbox_t> EngineInference(cv::Mat &image, std::vector<bbox_t> &bboxes);

    void postprocess(float *output, const std::vector<PoseEstimation::bbox> &cropped_boxes, std::vector<PoseEstimation::PoseKeypoints> &poseKeypoints);
    bool clearBuffer(bool freeInput = true, bool freeOutput = true);

private:
    bool forward_();
    const int IMAGE_WIDTH = 192;
    const int IMAGE_HEIGHT = 256;
    const int IMAGE_CHANNEL = 3;
    const int HEATMAP_CHANNEL = 17;
    const int HEATMAP_WIDTH = 48;
    const int HEATMAP_HEIGHT = 64;
    std::vector<int> person_index_class;
};

#endif // ALPHAPOSE_TRT_H