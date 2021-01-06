#include <iostream>
#include <memory>
#include <chrono>
#include <numeric>
#include <memory>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/c/builtin_op_data.h>

#ifdef GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <opencv2/opencv.hpp>

class FaceDetection
{
public:
    FaceDetection();
    ~FaceDetection();

    void detectFace(const cv::Mat &frame,
                    std::vector<cv::Rect> &boxes,
                    std::vector<float> &scores);

    void detectSingleFace(const cv::Mat &frame, cv::Rect &bbox);

private:
    void preprocess(const cv::Mat &frame);
    void postprocess(const cv::Mat &frame,
                     std::vector<cv::Rect> &boxes,
                     std::vector<float> &scores);

    float m_faceThreshold;
    float m_nmsThreshold;

    std::unique_ptr<tflite::Interpreter> m_interpreter;
    std::unique_ptr<tflite::FlatBufferModel> m_model;
};