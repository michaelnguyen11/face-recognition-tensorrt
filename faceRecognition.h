#include <iostream>
#include <numeric>
#include <vector>
#include <sys/stat.h>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/c/builtin_op_data.h>

#ifdef GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <opencv2/opencv.hpp>

const int g_mfnNumClass = 192;

class MobileFaceNet
{
public:
    MobileFaceNet();
    ~MobileFaceNet();

    std::array<float, g_mfnNumClass> extractFeatures(const cv::Mat &frame);

private:
    void preprocess(const cv::Mat &frame);
    void similarityCal();

    std::unique_ptr<tflite::Interpreter> m_interpreter;
    std::unique_ptr<tflite::FlatBufferModel> m_model;
};