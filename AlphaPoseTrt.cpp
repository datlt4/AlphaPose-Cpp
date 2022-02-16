#include "AlphaPoseTrt.h"

using namespace PoseEstimation;

bool AlphaPoseTRT::parseOnnxModel(const std::string &model_path, const int minBatchSize, const int optBatchSize, const int maxBatchSize, const int workspace)
{
    const char inputName[10] = "input";

    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    // We need to define explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> prediction_network{builder->createNetworkV2(explicitBatch)};
    // TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*prediction_network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        LOG(ERROR) << "ERROR: could not parse the model.";
        return false;
    }
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        LOG(ERROR) << "Create builder config failed.";
        return false;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(workspace);
    builder->setMaxBatchSize(maxBatchSize);
    // generate TensorRT engine optimized for the target platform
    nvinfer1::IOptimizationProfile *profileCalib = builder->createOptimizationProfile();
    // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
    profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{minBatchSize, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH});
    profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{optBatchSize, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH});
    profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH});
    config->addOptimizationProfile(profileCalib);
    this->prediction_engine.reset(builder->buildEngineWithConfig(*prediction_network, *config));
    this->prediction_context.reset(this->prediction_engine->createExecutionContext());
    return true;
}

bool AlphaPoseTRT::saveEngine(const std::string &fileName)
{
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        LOG(ERROR) << "Cannot open engine file: " << fileName;
        return false;
    }

    TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{this->prediction_engine->serialize()};
    if (serializedEngine == nullptr)
    {
        LOG(ERROR) << "Engine serialization failed";
        return false;
    }

    engineFile.write(static_cast<char *>(serializedEngine->data()), serializedEngine->size());

    return !engineFile.fail();
}

bool AlphaPoseTRT::loadEngine(const std::string &fileName)
{
    std::ifstream engineFile(fileName, std::ios::binary);
    if (!engineFile)
    {
        LOG(ERROR) << "Cannot open engine file: " << fileName;
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if (!engineFile.good())
    {
        LOG(ERROR) << "Error loading engine file";
        return false;
    }

    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    this->prediction_engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    this->prediction_context.reset(this->prediction_engine->createExecutionContext());
    return this->prediction_engine != nullptr;
}

std::vector<float> AlphaPoseTRT::prepareImage(cv::Mat &img, bbox &box, bbox &outBox)
{
    std::vector<float> result(long(IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL));
    float *data = result.data();
    cv::Mat rgbMat, transformed, transformed_255;
    cv::cvtColor(img, rgbMat, cv::ColorConversionCodes::COLOR_BGR2RGB);
    float aspect_ratio = 0.75;
    cv::Point_<float> center(box.rect.x + box.rect.width / 2, box.rect.y + box.rect.height / 2);
    float w, h;
    if (box.rect.width > (aspect_ratio * box.rect.height))
    {
        w = box.rect.width;
        h = w / aspect_ratio;
    }
    else if (box.rect.width < (aspect_ratio * box.rect.height))
    {
        h = box.rect.height;
        w = h * aspect_ratio;
    }
    cv::Point_<float> scale(w, h);
    if (center.x != -1)
    {
        scale = scale * 1.25; // x scale_mult
    }
    float src_w = scale.x;
    float dst_w = static_cast<float>(IMAGE_WIDTH);
    float dst_h = static_cast<float>(IMAGE_HEIGHT);
    float rot = 0;
    float rot_rad = M_PI * rot / 180.0;
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);
    cv::Point_<float> src_dir(0.0 * cs - (-0.5 * src_w) * sn, 0.0 * sn + (-0.5 * src_w) * cs);
    cv::Point_<float> dst_dir(0.0, -0.5 * dst_w);
    cv::Point_<float> direct_src(-src_dir);
    cv::Point_<float> src[3] = {center, center + src_dir, center + src_dir + cv::Point_<float>(-direct_src.y, direct_src.x)};
    cv::Point_<float> direct_dst(-dst_dir);
    cv::Point_<float> dst[3] = {cv::Point_<float>(dst_w * 0.5, dst_h * 0.5), cv::Point_<float>(dst_w * 0.5, dst_h * 0.5) + dst_dir, cv::Point_<float>(dst_w * 0.5, dst_h * 0.5) + dst_dir + cv::Point_<float>(-direct_dst.y, direct_dst.x)};
    cv::Mat trans = cv::getAffineTransform(src, dst);
    cv::warpAffine(rgbMat, transformed, trans, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), cv::INTER_LINEAR);
    float xmin = center.x - scale.x * 0.5;
    float ymin = center.y - scale.y * 0.5;
    outBox = bbox(xmin, ymin, scale.x, scale.y, 1.0);
    transformed.convertTo(transformed_255, CV_32F, 1.0 / 255);
    cv::subtract(transformed_255, cv::Scalar(0.406, 0.457, 0.480), transformed_255);
    // HWC TO CHW
    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 2),
    };
    cv::split(transformed_255, split_img);
    return result;
}

bool AlphaPoseTRT::processInput(float *hostDataBuffer, const int batchSize, cudaStream_t &stream)
{
    // std::vector< void* > input_buffers(this->prediction_engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < this->prediction_engine->getNbBindings(); ++i)
    {
        int32_t binding_size = PoseEstimation::volume(this->prediction_engine->getBindingDimensions(i)) * batchSize * sizeof(float);
        binding_size = (binding_size > 0) ? binding_size : -binding_size;
        // std::cout << "Size of: " << binding_size << std::endl;
        if (this->prediction_engine->bindingIsInput(i))
        {
            input_buffers.emplace_back(new float());
            cudaMalloc(&input_buffers.back(), binding_size);
            prediction_input_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
        else
        {
            output_buffers.emplace_back(new float());
            cudaMalloc(&output_buffers.back(), binding_size);
            prediction_output_dims.emplace_back(this->prediction_engine->getBindingDimensions(i));
        }
    }

    if (prediction_input_dims.empty() || prediction_output_dims.empty())
    {
        LOG(ERROR) << "Expect at least one input and one output for network";
        return false;
    }

    float *gpu_input_0 = (float *)input_buffers[0];

    // TensorRT copy way
    // Host memory for input buffer
    if (cudaMemcpyAsync(gpu_input_0, hostDataBuffer, size_t(batchSize * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL * sizeof(float)), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        LOG(ERROR) << "Input corrupted or CUDA error, abort ";
        return false;
    }

    return true;
}

std::vector<bbox_t> AlphaPoseTRT::EngineInference(cv::Mat &image, std::vector<bbox_t> &bboxes)
{
    int nBBoxes = bboxes.size();
    int maxBatchSize = this->prediction_engine->getMaxBatchSize();
    int nBatches = nBBoxes / maxBatchSize + ((nBBoxes % maxBatchSize != 0) ? 1 : 0);
    if (!(nBBoxes > 0))
        return std::vector<bbox_t>{};

    // cv::Mat image = imageVizgard.get_raw_frame_host();
    std::vector<int> indices;
    std::vector<bbox> person_bboxes;
    std::vector<bbox_t> output;

    for (int idx = 0; idx < bboxes.size(); ++idx)
    {
        bool is_person = false;
        for (int i : this->person_index_class)
        {
            if (bboxes.at(idx).obj_id == i)
            {
                is_person = true;
                break;
            }
        }
        if (is_person)
        {
            indices.push_back(idx);
            person_bboxes.push_back(bbox(bboxes.at(idx).x, bboxes.at(idx).y, bboxes.at(idx).w, bboxes.at(idx).h, bboxes.at(idx).prob));
        }
        else
        {
            output.push_back(bboxes.at(idx));
        }
    }

    for (int i = 0; i < nBatches; ++i)
    {
        int batchSize = (i == (nBatches - 1)) ? (nBBoxes - maxBatchSize * i) : maxBatchSize;
        std::vector<PoseKeypoints> poseKeypoints;
        std::vector<bbox> cropped_boxes;
        // std::vector<float> curInput(long(IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL * batchSize));
        std::vector<float> curInput{};
        std::vector<PoseEstimation::bbox>::iterator it;

        // for (bbox &objBox : person_bboxes)
        for (it = person_bboxes.begin() + maxBatchSize * i; it != person_bboxes.begin() + maxBatchSize * i + batchSize; ++it)
        {
            bbox processedBox;
            std::vector<float> imageData = this->prepareImage(image, *it, processedBox);
            curInput.insert(curInput.end(), imageData.begin(), imageData.end());
            cropped_boxes.push_back(processedBox);
        }
        this->processInput(curInput.data(), batchSize, stream);
        std::vector<void *> predicitonBindings = {(float *)input_buffers[0], (float *)output_buffers[0]};
        // LOG(INFO) << "Input " << log_cuda_bf(prediction_input_dims[0], predicitonBindings[0], 100);
        this->prediction_context->setBindingDimensions(0, nvinfer1::Dims4(batchSize, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH));
        this->prediction_context->enqueue(batchSize, predicitonBindings.data(), 0, nullptr);
        // LOG(INFO) << "Output: " << log_cuda_bf(prediction_output_dims[0], predicitonBindings[1], 100);
        std::vector<float> hmOutput(batchSize * HEATMAP_CHANNEL * HEATMAP_HEIGHT * HEATMAP_WIDTH);
        cudaMemcpy(hmOutput.data(), predicitonBindings[1], hmOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);
        postprocess(hmOutput.data(), cropped_boxes, poseKeypoints);
        for (int idx = maxBatchSize * i, c = 0; idx < maxBatchSize * i + batchSize; ++idx, ++c)
        {
            bboxes.at(indices.at(idx)).keypoints = poseKeypoints.at(c).keypoints;
            bboxes.at(indices.at(idx)).kp_scores = poseKeypoints.at(c).kp_scores;
            output.push_back(bboxes.at(indices.at(idx)));
        }
        clearBuffer();
    }
    return output;
};

void AlphaPoseTRT::postprocess(float *output, const std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints)
{
    for (int i = 0; i < cropped_boxes.size(); ++i)
    {
        float *out = output;
        cv::Mat result_matrix = cv::Mat(HEATMAP_CHANNEL, HEATMAP_WIDTH * HEATMAP_HEIGHT, CV_32FC1, out + i * HEATMAP_CHANNEL * HEATMAP_WIDTH * HEATMAP_HEIGHT);
        PoseKeypoints kps;
        std::vector<int> index;
        std::vector<cv::Point_<float>> coords;
        for (int row_num = 0; row_num < HEATMAP_CHANNEL; row_num++)
        {
            float *row = result_matrix.ptr<float>(row_num);
            float *max_pos = std::max_element(row, row + HEATMAP_WIDTH * HEATMAP_HEIGHT);
            int index = max_pos - row;
            int _x = index % 48;
            int _y = index / 48;
            cv::Point_<float> p(static_cast<float>(_x), static_cast<float>(_y));
            if ((1 < _x < 47) && (1 < _y < 63))
            {
                cv::Point_<float> diff(row[_x + 1 + HEATMAP_WIDTH * _y] - row[_x - 1 + HEATMAP_WIDTH * _y], row[_x + HEATMAP_WIDTH * (_y + 1)] - row[_x + HEATMAP_WIDTH * (_y - 1)]);
                p += cv::Point_<float>((diff.x == 0) ? 0.0 : ((diff.x < 0) ? -0.25 : 0.25), (diff.y == 0) ? 0.0 : ((diff.y < 0) ? -0.25 : 0.25));
            }
            kps.kp_scores.push_back(row[index]);
            coords.push_back(p);
        }

        // transform bbox to scale
        bbox box = cropped_boxes.at(i);
        cv::Point_<float> center(box.rect.x + box.rect.width / 2, box.rect.y + box.rect.height / 2);
        cv::Point_<float> scale(box.rect.width, box.rect.height);

        // Transform back
        for (int i = 0; i < coords.size(); i++)
        {
            float src_w = scale.x;
            float dst_w = 48.0;
            float dst_h = 64.0;
            float rot = 0.0;
            float rot_rad = M_PI * rot / 180.0;
            float sn = sin(rot_rad);
            float cs = cos(rot_rad);
            cv::Point_<float> src_dir(0.0 * cs - (-0.5 * src_w) * sn, 0.0 * sn + (-0.5 * src_w) * cs);
            cv::Point_<float> dst_dir(0.0, -0.5 * dst_w);
            cv::Point_<float> direct_src(-src_dir);
            cv::Point_<float> src[3] = {center, center + src_dir, center + src_dir + cv::Point_<float>(-direct_src.y, direct_src.x)};
            cv::Point_<float> direct_dst(-dst_dir);
            cv::Point_<float> dst[3] = {cv::Point_<float>(dst_w * 0.5, dst_h * 0.5), cv::Point_<float>(dst_w * 0.5, dst_h * 0.5) + dst_dir, cv::Point_<float>(dst_w * 0.5, dst_h * 0.5) + dst_dir + cv::Point_<float>(-direct_dst.y, direct_dst.x)};
            cv::Mat trans = cv::getAffineTransform(dst, src);
            cv::Point_<float> target_coords(trans.at<double>(0, 0) * coords[i].x + trans.at<double>(0, 1) * coords[i].y + trans.at<double>(0, 2),
                                            trans.at<double>(1, 0) * coords[i].x + trans.at<double>(1, 1) * coords[i].y + trans.at<double>(1, 2));
            kps.keypoints.push_back(target_coords);
        }
        poseKeypoints.push_back(kps);
    }
}

bool AlphaPoseTRT::clearBuffer(bool freeInput, bool freeOutput)
{
    // LOGCUDABUFFER(prediction_output_dims[0], this->output_buffers[0], 100);
    this->prediction_input_dims.clear();
    this->prediction_output_dims.clear();
    try
    {
        if (freeInput)
            for (void *buf : input_buffers)
                cudaFree(buf);

        if (freeOutput)
            for (void *buf : output_buffers)
                cudaFree(buf);
    }
    catch (std::runtime_error &e)
    {
        LOG(ERROR) << e.what() << std::endl;
        return false;
    }
    input_buffers.clear();
    output_buffers.clear();
    // TODO: Properly re wrote this
    return true;
}

int64_t PoseEstimation::volume(const nvinfer1::Dims &d)
{
    int64_t result = std::accumulate(d.d, d.d + d.nbDims, 1, [](int32_t x, int32_t y)
                                     { return x * y; });
    return result;
}
