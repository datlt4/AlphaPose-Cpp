#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

struct bbox_t
{
    bbox_t() {}
    bbox_t(float x, float y, float w, float h, float prob, int obj_id) : x(x), y(y), w(w), h(h), prob(prob), obj_id(obj_id) {}
    float x, y, w, h;  // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;        // confidence - probability that the object was found correctly
    int obj_id;        // class of object - from range [0, classes-1]
    int track_id = -1; // tracking id for video (0 - untracked, 1 - inf - tracked object)
    std::vector<cv::Point2f> keypoints = {};
    std::vector<float> kp_scores; // proposal_score

    friend std::ostream &operator<<(std::ostream &os, const bbox_t &bb)
    {
        os << "[ X ]: " << bb.x << "\t[ Y ]: " << bb.y << "\t[ W ]: " << bb.w << "\t[ H ]: " << bb.h << "\t[ PROB ]: " << bb.prob << "\t[ OBJ_ID ]: " << bb.obj_id << "\t[ TRACK_ID ]: " << bb.track_id << std::endl;
        return os;
    }
};
