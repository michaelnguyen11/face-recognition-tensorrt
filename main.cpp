#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <NvInferRuntimeCommon.h>

#include "FaceDetectionYolov5.h"

static Logger gLogger;

int main(int argc, const char **argv)
{
    cudaSetDevice(0);

    std::string modelType = "yolov5s";
    std::string inputName = "data";
    std::string outputName = "prob";

    float nmsThreshold = 0.4;
    float confidentThreshold = 0.9;

    std::unique_ptr<FaceDetectionYolov5> yolov5 = std::make_unique<FaceDetectionYolov5>(modelType, inputName, outputName);

    // Yolov5 parameters

    // assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
    static const int yoloOutputSize = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    static const int batchSize = 1;
    char *trtModelStream{nullptr};
    size_t modelSize;

    // Build engine file
    yolov5->buildEngine(batchSize);
    yolov5->loadEngine(&trtModelStream, modelSize);

    // prepare input data ---------------------------
    static float inputData[batchSize * 3 * Yolo::INPUT_H * Yolo::INPUT_W];

    static float probability[batchSize * yoloOutputSize];
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, modelSize);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(inputName.c_str());
    const int outputIndex = engine->getBindingIndex(outputName.c_str());
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * yoloOutputSize * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        cv::Mat preprocessOut = yolov5->preProcessing(frame);
        // copy image data to float data
        for (int i = 0; i < Yolo::INPUT_H * Yolo::INPUT_W; i++)
        {
            inputData[i] = preprocessOut.at<cv::Vec3b>(i)[2] / 255.0;
            inputData[i + Yolo::INPUT_H * Yolo::INPUT_W] = preprocessOut.at<cv::Vec3b>(i)[1] / 255.0;
            inputData[i + 2 * Yolo::INPUT_H * Yolo::INPUT_W] = preprocessOut.at<cv::Vec3b>(i)[0] / 255.0;
        }
        // Run inference
        yolov5->doInference(*context, stream, buffers, inputData, probability, batchSize);
        std::vector<std::vector<Yolo::Detection>> batchResults(1);
        yolov5->nms(batchResults[0], &probability[0], confidentThreshold, nmsThreshold);

        for (size_t j = 0; j < batchResults[0].size(); j++)
        {
            cv::Rect r = yolov5->getRectangle(frame, batchResults[0][j].bbox);
            cv::rectangle(frame, r, cv::Scalar(255, 255, 0), 2);
            cv::putText(frame, std::to_string((int)batchResults[0][j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            std::cout << "confident: " << batchResults[0][j].conf << std::endl;
        }
        cv::imshow("current frame", frame);
        if (cv::waitKey(27) >= 0)
            break;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}