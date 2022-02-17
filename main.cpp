#include "AlphaPoseTrt.h"

alphaposeTrtLoger::Logger *allogger = alphaposeTrtLoger::LoggerFactory::CreateConsoleLogger(alphaposeTrtLoger::INFO);
#ifdef BUILD_TRTEXEC
// AlphaPoseTrt.exe --onnx ..\..\model-zoo\fast_pose_res50\fast_res50_256x192_dynamic.onnx --engine ..\..\model-zoo\fast_pose_res50\fast_res50_256x192_fp16_dynamic_wins.engine --minBatchSize 1 --optBatchSize 8 --maxBatchSize 32 --dynamic
struct TrtexecConfig
{
    TrtexecConfig() : minBatchSize{1}, optBatchSize{1}, maxBatchSize{1}, workspace{1ULL << 30}, dynamic{true} {}
    std::string onnx_dir;
    std::string engine_dir;
    bool dynamic;
    int minBatchSize;
    int optBatchSize;
    int maxBatchSize;
    std::size_t workspace;
    friend std::ostream &operator<<(std::ostream &os, const TrtexecConfig config)
    {
        os << "  --onnx        : " << config.onnx_dir << std::endl
           << "  --engine      : " << config.engine_dir << std::endl
           << "  --dynamic     : " << (config.dynamic ? "True" : "False") << std::endl
           << "  --minBatchSize: " << config.minBatchSize << std::endl
           << "  --optBatchSize: " << config.optBatchSize << std::endl
           << "  --maxBatchSize: " << config.maxBatchSize << std::endl;
        return os;
    }
};

void ShowHelpAndExit(const char *szBadOption);
bool ParseCommandLine(int argc, char *argv[], TrtexecConfig &config);
int main(int argc, char **argv)
{
    TrtexecConfig config;
    if (ParseCommandLine(argc, argv, config))
    {
        std::unique_ptr<AlphaPoseTRT> pose_est = std::make_unique<AlphaPoseTRT>(std::vector<int>{1});
        if (config.dynamic)
        {
            pose_est->parseOnnxModel(config.onnx_dir, config.minBatchSize, config.optBatchSize, config.maxBatchSize, config.workspace);
            pose_est->saveEngine(config.engine_dir);
        }
        LOG(INFO) << "[ PASSED ]:\n" << config;
    }
    else
        LOG(ERROR) << "[ ERROR ] STOP!!!";
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "    --onnx [PATH]       : path to Onnx file" << std::endl
        << "    --engine [PATH]     : name of output Engine file" << std::endl
        << "    --dynamic           : indicate that build engine with Dynamic Batch Size" << std::endl
        << "    --minBatchSize [Int]: min batchsize" << std::endl
        << "    --optBatchSize [Int]: optimization batchsize" << std::endl
        << "    --maxBatchSize [Int]: max batchsize" << std::endl
        << "    --workspace [Int]   : max workspace size in MB" << std::endl;

    oss << std::endl;

    if (bThrowError)
        throw std::invalid_argument(oss.str());
    else
        std::cout << oss.str();
}

bool ParseCommandLine(int argc, char *argv[], TrtexecConfig &config)
{
    if (argc <= 1)
    {
        ShowHelpAndExit();
        return false;
    }
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == std::string("--help"))
        {
            ShowHelpAndExit();
            return false;
        }
        else if (std::string(argv[i]) == std::string("--onnx"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--onnx");
                return false;
            }

            else
                config.onnx_dir = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--engine"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--engine");
                return false;
            }
            else
                config.engine_dir = std::string(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--dynamic"))
        {
            config.dynamic = true;
        }
        else if (std::string(argv[i]) == std::string("--minBatchSize"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--minBatchSize");
                return false;
            }
            else
                config.minBatchSize = std::stoi(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--optBatchSize"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--optBatchSize");
                return false;
            }
            else
                config.optBatchSize = std::stoi(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--maxBatchSize"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--maxBatchSize");
                return false;
            }
            else
                config.maxBatchSize = std::stoi(argv[i]);
            continue;
        }
        else if (std::string(argv[i]) == std::string("--workspace"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("--workspace");
                return false;
            }
            else
                config.maxBatchSize = std::stoi(argv[i]) * (1ULL << 20);
            continue;
        }
        else
        {
            {
                ShowHelpAndExit((std::string("input not include ") + std::string(argv[i])).c_str());
                return false;
            }
        }
    }
    return true;
}

#else  // BUILD_TRTEXEC
int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("pose2.jpg");
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

    std::cout << "[ ALPHAPOSE ][ TENSORRT ]" << std::endl;
    std::unique_ptr<AlphaPoseTRT> pose_est = std::make_unique<AlphaPoseTRT>(std::vector<int>{1});
    pose_est->loadEngine("model-zoo\\fast_pose_res50\\fast_res50_256x192_fp16_dynamic_wins.engine");

    for (int i = 0; i < 100; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = pose_est->EngineInference(image, result);
        if (i == 0)
        {
            for (bbox_t &bb : result)
            {
                PoseEstimation::draw(image, bb);
            }
            cv::imwrite("result_tensorrt.jpg", image);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "[ LOG ][ TensorRT ] duration: " << duration.count() << "ms" << std::endl;
    }
}
#endif // BUILD_TRTEXEC
