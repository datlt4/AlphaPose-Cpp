#include "AlphaPose.h"

AlphaPose::AlphaPose(std::string model_path, std::vector<int> person_index_class) : person_index_class{person_index_class}
{
    try
    {
        this->al = torch::jit::load(model_path, torch::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << e.msg() << std::endl;
        std::cerr << "[Error] loading the model\n";
        assert(false);
    }
}

void AlphaPose::preprocess(cv::Mat &img, bbox &box, torch::Tensor &imageTensor, bbox &outBox)
{
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
    float dst_w = 192.0;
    float dst_h = 256.0;
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
    cv::warpAffine(rgbMat, transformed, trans, cv::Size(192, 256), cv::INTER_LINEAR);
    // cv::imwrite("transformed_00.jpg", transformed);
    float xmin = center.x - scale.x * 0.5;
    float ymin = center.y - scale.y * 0.5;
    outBox = bbox(xmin, ymin, scale.x, scale.y, 1.0);
    transformed.convertTo(transformed_255, CV_32F, 1.0 / 255);
    cv::Mat channels[3];
    cv::split(transformed_255, channels);
    channels[0] = (channels[0] - 0.406);
    channels[1] = (channels[1] - 0.457);
    channels[2] = (channels[2] - 0.480);
    cv::merge(channels, 3, transformed_255);

    imageTensor = torch::from_blob(
                      transformed_255.data,
                      {1, transformed_255.rows, transformed_255.cols, 3}, torch::TensorOptions(torch::kFloat))
                      .permute({0, 3, 1, 2})
                      .toType(torch::kFloat)
                      .cuda();
}

void AlphaPose::heatmap_to_coord(const torch::Tensor &hms, const bbox &box, PoseKeypoints &preds)
{
    std::tuple<torch::Tensor, torch::Tensor> mm = torch::max(hms.reshape({17, -1}), 1);
    torch::Tensor maxvals = std::get<0>(mm); // kFloat
    torch::Tensor idx = std::get<1>(mm).to(torch::kInt32);
    std::vector<cv::Point_<float>> coords;
    int *idx_ptr = (int *)idx.data_ptr();
    float *maxval_ptr = (float *)maxvals.data_ptr();
    for (int idx = 0; idx < maxvals.sizes()[0]; idx++)
    {
        cv::Point_<float> p((float)((*idx_ptr) % 48), (float)((*idx_ptr) / 48));
        coords.push_back(p);
        preds.kp_scores.push_back(*maxval_ptr);
        idx_ptr++;
        maxval_ptr++;
    }
    for (int p = 0; p < coords.size(); p++)
    {
        torch::Tensor hm = hms.slice(1, p, p + 1);
        if ((1 < coords[p].x < 47) || (1 < coords[p].y < 63))
        {
            cv::Point_<float> diff(*((float *)hm.data_ptr() + (int)coords[p].x + 1 + 48 * (int)coords[p].y) - *((float *)hm.data_ptr() + (int)coords[p].x - 1 + 48 * (int)coords[p].y),
                                   *((float *)hm.data_ptr() + (int)coords[p].x + 48 * ((int)coords[p].y + 1)) - *((float *)hm.data_ptr() + (int)coords[p].x + 48 * ((int)coords[p].y - 1)));
            coords[p] += cv::Point_<float>((diff.x == 0) ? 0.0 : ((diff.x < 0) ? -0.25 : 0.25), (diff.y == 0) ? 0.0 : ((diff.y < 0) ? -0.25 : 0.25));
        }
    }

    // transform bbox to scale
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
        preds.keypoints.push_back(target_coords);
    }
}

void AlphaPose::postprocess(const torch::Tensor &hm_data, const std::vector<bbox> &cropped_boxes, std::vector<PoseKeypoints> &poseKeypoints)
{
    for (int i = 0; i < hm_data.sizes()[0]; i++)
    {
        PoseKeypoints pKp;
        std::vector<float> pose_score;
        torch::Tensor hms = hm_data.slice(0, i, i + 1);
        heatmap_to_coord(hms, cropped_boxes[i], pKp);

        poseKeypoints.push_back(pKp);
    }
}

// std::vector<bbox_t> AlphaPose::predict(VizgardFrame& imageVizgard, std::vector<bbox_t>& bboxes)
std::vector<bbox_t> AlphaPose::predict(cv::Mat &image, std::vector<bbox_t> &bboxes)
{
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
    std::vector<PoseKeypoints> poseKeypoints;
    this->predict(image, person_bboxes, poseKeypoints);
    for (int i = 0; i < indices.size(); ++i)
    {
        bboxes.at(indices.at(i)).keypoints = poseKeypoints.at(i).keypoints;
        bboxes.at(indices.at(i)).kp_scores = poseKeypoints.at(i).kp_scores;
        output.push_back(bboxes.at(indices.at(i)));
    }
    return output;
}

void AlphaPose::predict(cv::Mat &image, std::vector<bbox> &objBoxes, std::vector<PoseKeypoints> &poseKeypoints)
{
    cv::Mat processedImage;
    std::vector<bbox> cropped_boxes;
    std::vector<torch::Tensor> imageBatch;
    for (bbox objBox : objBoxes)
    {
        bbox processedBox;
        torch::Tensor imageTensor;
        preprocess(image, objBox, imageTensor, processedBox);
        cropped_boxes.push_back(processedBox);
        imageBatch.push_back(imageTensor);
    }

    if (objBoxes.size() > 0)
    {
        torch::Tensor iimageBatch = torch::cat(imageBatch, 0);
        this->al.forward({iimageBatch});
        torch::Tensor hm = this->al.forward({iimageBatch}).toTensor().to(torch::kCPU);
        postprocess(hm, cropped_boxes, poseKeypoints);
    }
}
