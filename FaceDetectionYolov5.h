#ifndef FACE_DETECTION_YOLOV5_H_
#define FACE_DETECTION_YOLOV5_H_

#include <chrono>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "Logging.h"
#include "YoloLayer.h"
#include "Utils.h"

class FaceDetectionYolov5
{
public:
    explicit FaceDetectionYolov5(std::string modelType, std::string inputName, std::string outputName);
    ~FaceDetectionYolov5();

    cv::Mat preProcessing(const cv::Mat &image);

    void nms(std::vector<Yolo::Detection> &result, float *output, float confThreshold, float nmsThreshold = 0.5);

    void loadEngine(char** trtModelStream, size_t &size);
    void buildEngine(int maxBatchSize);

    void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers,
                    float* input, float* output, int batchSize);
    cv::Rect getRectangle(const cv::Mat &image, float bbox[4]);

private:
    static Logger tensorRTLogger;

    nvinfer1::ICudaEngine* createYoloV5Engine(int maxBatchSize, nvinfer1::IBuilder* builder,
                                              nvinfer1::IBuilderConfig* config, nvinfer1::DataType dataType);

    std::map<std::string, nvinfer1::Weights> loadWeight(const std::string &weightPath);

    float iou(float lbox[4], float rbox[4]);

    std::string _modelType;
    std::string _engineFileName;
    std::string _weightPath;
    std::string _inputName;
    std::string _outputName;
    float _widthMultiple;
    float _depthMultiple;

    // TensorRT Yolo used layers
    nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition* network,
                                          std::map<std::string, nvinfer1::Weights>& weightMap,
                                          nvinfer1::ITensor& input,
                                          std::string layerName, float eps);

    nvinfer1::ILayer* convBlock(nvinfer1::INetworkDefinition* network,
                                std::map<std::string, nvinfer1::Weights>& weightMap,
                                nvinfer1::ITensor& input,
                                int outFeatureMap, int kernelSize, int stride, int numGroups,
                                std::string layerName);

    nvinfer1::ILayer* focus(nvinfer1::INetworkDefinition* network,
                            std::map<std::string, nvinfer1::Weights>& weightMap,
                            nvinfer1::ITensor& input,
                            int inputChannels, int outputChannels,
                            int kernelSize, std::string layerName);

    nvinfer1::ILayer* bottleneck(nvinfer1::INetworkDefinition *network,
                                 std::map<std::string, nvinfer1::Weights>& weightMap,
                                 nvinfer1::ITensor& input,
                                 int c1, int c2, bool shortcut,
                                 int g, float e, std::string layerName);

    nvinfer1::ILayer* bottleneckCSP(nvinfer1::INetworkDefinition* network,
                                    std::map<std::string, nvinfer1::Weights>& weightMap,
                                    nvinfer1::ITensor& input,
                                    int c1, int c2, int n, bool shortcut,
                                    int g, float e, std::string layerName);

    nvinfer1::ILayer* SPP(nvinfer1::INetworkDefinition *network,
                          std::map<std::string, nvinfer1::Weights>& weightMap,
                          nvinfer1::ITensor& input,
                          int c1, int c2, int k1, int k2, int k3,
                          std::string layerName);

    nvinfer1::IDeconvolutionLayer* deconvBlock(nvinfer1::INetworkDefinition* network,
                                               std::map<std::string, nvinfer1::Weights>& weightMap,
                                               nvinfer1::ITensor& input,
                                               int k, std::string layerName);

};


#endif // !FACE_DETECTION_YOLOV5_H_
