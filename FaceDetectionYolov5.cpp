#include "FaceDetectionYolov5.h"

Logger FaceDetectionYolov5::tensorRTLogger;

FaceDetectionYolov5::FaceDetectionYolov5(std::string modelType, std::string inputName, std::string outputName)
    : _modelType(modelType), _inputName(inputName), _outputName(outputName)
{
    if (_modelType == "yolov5s")
    {
        _widthMultiple = 0.5;
        _depthMultiple = 0.33;
    }
    else if (_modelType == "yolov5m")
    {
        _widthMultiple = 0.75;
        _depthMultiple = 0.67;
    }
    else if (_modelType == "yolov5l")
    {
        _widthMultiple = 1.0;
        _depthMultiple = 1.0;
    }
    else
    {
        std::cerr << "Don't support model type: " << _modelType << std::endl;
    }

    std::cout << "Using Yolov5 model type: " << _modelType << std::endl;
    _engineFileName = "data/" + _modelType + "_widerface.engine";
    _weightPath = "data/" + _modelType + "_widerface.wts";
}

FaceDetectionYolov5::~FaceDetectionYolov5() {}

cv::Mat FaceDetectionYolov5::preProcessing(const cv::Mat &image)
{
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (image.cols * 1.0);
    float r_h = Yolo::INPUT_H / (image.rows * 1.0);
    if (r_h > r_w)
    {
        w = Yolo::INPUT_W;
        h = r_w * image.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    }
    else
    {
        w = r_h * image.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat resizedImage(h, w, CV_8UC3);
    cv::resize(image, resizedImage, resizedImage.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    resizedImage.copyTo(out(cv::Rect(x, y, resizedImage.cols, resizedImage.rows)));

    return out;
}

cv::Rect FaceDetectionYolov5::getRectangle(const cv::Mat &image, float bbox[4])
{
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (image.cols * 1.0);
    float r_h = Yolo::INPUT_H / (image.rows * 1.0);
    if (r_h > r_w)
    {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * image.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * image.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * image.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * image.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

float FaceDetectionYolov5::iou(float lbox[4], float rbox[4])
{
    float interBox[] = {
        std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
        std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
        std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
        std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;
    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void FaceDetectionYolov5::nms(std::vector<Yolo::Detection> &result, float *output, float confThreshold, float nmsThreshold)
{
    int detectionSize = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; ++i)
    {
        if (output[1 + detectionSize * i + 4] < confThreshold)
            continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + detectionSize * i], detectionSize * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Yolo::Detection>());

        m[det.class_id].push_back(det);
    }

    for (auto it = m.begin(); it != m.end(); ++it)
    {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(),
                  [&](const Yolo::Detection &a, const Yolo::Detection &b) {
                      return a.conf > b.conf;
                  });

        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            result.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (this->iou(item.bbox, dets[n].bbox) > nmsThreshold)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void FaceDetectionYolov5::doInference(nvinfer1::IExecutionContext &context, cudaStream_t &stream, void **buffers,
                                      float *input, float *output, int batchSize)
{
    // assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
    const int outputSize = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void FaceDetectionYolov5::buildEngine(int maxBatchSize)
{
    std::ifstream file(_engineFileName, std::ios::binary);
    if (file.good())
    {
        std::cout << "Engine file have already existed. Start deserializing..." << std::endl;
        return;
    }

    nvinfer1::IHostMemory *modelStream{nullptr};
    // Create builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(tensorRTLogger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine *engine = createYoloV5Engine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT);

    assert(engine != nullptr);

    // Serialize the engine
    modelStream = engine->serialize();
    assert(modelStream != nullptr);
    std::ofstream p(_engineFileName, std::ios::binary);
    if (!p)
    {
        std::cerr << "Could not open plan output file" << std::endl;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    modelStream->destroy();
    engine->destroy();
    config->destroy();
    builder->destroy();
}

void FaceDetectionYolov5::loadEngine(char **trtModelStream, size_t &size)
{
    try
    {
        std::ifstream file(_engineFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            (*trtModelStream) = new char[size];
            assert(*trtModelStream);
            file.read(*trtModelStream, size);
            file.close();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to deserialize engine file: " << e.what() << '\n';
    }
}

nvinfer1::ICudaEngine *FaceDetectionYolov5::createYoloV5Engine(int maxBatchSize,
                                                               nvinfer1::IBuilder *builder,
                                                               nvinfer1::IBuilderConfig *config,
                                                               nvinfer1::DataType dataType)
{
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    nvinfer1::ITensor *data = network->addInput(_inputName.c_str(), dataType, nvinfer1::Dims3{3, Yolo::INPUT_H, Yolo::INPUT_W});
    assert(data);

    std::map<std::string, nvinfer1::Weights> weightMap = this->loadWeight(_weightPath);
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};

    /* ------ yolov5 backbone------ */
    auto focus0 = this->focus(network, weightMap, *data, 3, static_cast<int>(64 * _widthMultiple), 3, "model.0");
    auto conv1 = this->convBlock(network, weightMap, *focus0->getOutput(0), static_cast<int>(128 * _widthMultiple), 3, 2, 1, "model.1");
    auto bottleneck_csp2 = this->bottleneckCSP(network, weightMap, *conv1->getOutput(0), static_cast<int>(128 * _widthMultiple), static_cast<int>(128 * _widthMultiple), static_cast<int>(3 * _depthMultiple), true, 1, 0.5, "model.2");
    auto conv3 = this->convBlock(network, weightMap, *bottleneck_csp2->getOutput(0), static_cast<int>(256 * _widthMultiple), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = this->bottleneckCSP(network, weightMap, *conv3->getOutput(0), static_cast<int>(256 * _widthMultiple), static_cast<int>(256 * _widthMultiple), static_cast<int>(9 * _depthMultiple), true, 1, 0.5, "model.4");
    auto conv5 = this->convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), static_cast<int>(512 * _widthMultiple), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = this->bottleneckCSP(network, weightMap, *conv5->getOutput(0), static_cast<int>(512 * _widthMultiple), static_cast<int>(512 * _widthMultiple), static_cast<int>(9 * _depthMultiple), true, 1, 0.5, "model.6");
    auto conv7 = this->convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), static_cast<int>(1024 * _widthMultiple), 3, 2, 1, "model.7");
    auto spp8 = this->SPP(network, weightMap, *conv7->getOutput(0), static_cast<int>(1024 * _widthMultiple), static_cast<int>(1024 * _widthMultiple), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = this->bottleneckCSP(network, weightMap, *spp8->getOutput(0), static_cast<int>(1024 * _widthMultiple), static_cast<int>(1024 * _widthMultiple), static_cast<int>(3 * _depthMultiple), false, 1, 0.5, "model.9");
    auto conv10 = this->convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), static_cast<int>(512 * _widthMultiple), 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * static_cast<int>(512 * _widthMultiple) * 2 * 2));
    for (int i = 0; i < static_cast<int>(512 * _widthMultiple) * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }

    nvinfer1::Weights deconvwts11{nvinfer1::DataType::kFLOAT, deval, static_cast<int>(512 * _widthMultiple) * 2 * 2};
    nvinfer1::IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), static_cast<int>(512 * _widthMultiple), nvinfer1::DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(nvinfer1::DimsHW{2, 2});
    deconv11->setNbGroups(static_cast<int>(512 * _widthMultiple));
    weightMap["deconv11"] = deconvwts11;

    nvinfer1::ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = this->bottleneckCSP(network, weightMap, *cat12->getOutput(0), static_cast<int>(1024 * _widthMultiple), static_cast<int>(512 * _widthMultiple), static_cast<int>(3 * _depthMultiple), false, 1, 0.5, "model.13");
    auto conv14 = this->convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), static_cast<int>(256 * _widthMultiple), 1, 1, 1, "model.14");

    nvinfer1::Weights deconvwts15{nvinfer1::DataType::kFLOAT, deval, static_cast<int>(256 * _widthMultiple) * 2 * 2};
    nvinfer1::IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), static_cast<int>(256 * _widthMultiple), nvinfer1::DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(nvinfer1::DimsHW{2, 2});
    deconv15->setNbGroups(static_cast<int>(256 * _widthMultiple));
    nvinfer1::ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = this->bottleneckCSP(network, weightMap, *cat16->getOutput(0), static_cast<int>(512 * _widthMultiple), static_cast<int>(256 * _widthMultiple), static_cast<int>(3 * _depthMultiple), false, 1, 0.5, "model.17");

    // yolo layer 0
    nvinfer1::IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = this->convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), static_cast<int>(256 * _widthMultiple), 3, 2, 1, "model.18");
    nvinfer1::ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = this->bottleneckCSP(network, weightMap, *cat19->getOutput(0), static_cast<int>(512 * _widthMultiple), static_cast<int>(512 * _widthMultiple), static_cast<int>(3 * _depthMultiple), false, 1, 0.5, "model.20");

    //yolo layer 1
    nvinfer1::IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = this->convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), static_cast<int>(512 * _widthMultiple), 3, 2, 1, "model.21");
    nvinfer1::ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = this->bottleneckCSP(network, weightMap, *cat22->getOutput(0), static_cast<int>(1024 * _widthMultiple), static_cast<int>(1024 * _widthMultiple), static_cast<int>(3 * _depthMultiple), false, 1, 0.5, "model.23");

    nvinfer1::IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), nvinfer1::DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const nvinfer1::PluginFieldCollection *pluginData = creator->getFieldNames();
    nvinfer1::IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    nvinfer1::ITensor *inputTensors_yolo[] = {det2->getOutput(0), det1->getOutput(0), det0->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(_outputName.c_str());
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> FaceDetectionYolov5::loadWeight(const std::string &weightPath)
{
    std::map<std::string, nvinfer1::Weights> weightMap;

    std::ifstream input(weightPath);
    assert(input.is_open() && "Unable to load weight file");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0; x < size; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

nvinfer1::IScaleLayer *FaceDetectionYolov5::addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                                           std::map<std::string, nvinfer1::Weights> &weightMap,
                                                           nvinfer1::ITensor &input, std::string layerName, float eps)
{
    float *gamma = (float *)weightMap[layerName + ".weight"].values;
    float *beta = (float *)weightMap[layerName + ".bias"].values;
    float *mean = (float *)weightMap[layerName + ".running_mean"].values;
    float *var = (float *)weightMap[layerName + ".running_var"].values;
    int len = weightMap[layerName + ".running_var"].count;

    float *scaleValue = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scaleValue[i] = gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scaleValue, len};

    float *shiftValue = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shiftValue[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shiftValue, len};

    float *powerValue = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        powerValue[i] = 1.0;
    }
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, powerValue, len};

    weightMap[layerName + ".scale"] = scale;
    weightMap[layerName + ".shift"] = shift;
    weightMap[layerName + ".power"] = power;
    nvinfer1::IScaleLayer *scale_1 = network->addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

nvinfer1::ILayer *FaceDetectionYolov5::convBlock(nvinfer1::INetworkDefinition *network,
                                                 std::map<std::string, nvinfer1::Weights> &weightMap,
                                                 nvinfer1::ITensor &input,
                                                 int outFeatureMap, int kernelSize, int stride,
                                                 int numGroups, std::string layerName)
{
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    int padding = kernelSize / 2;
    nvinfer1::IConvolutionLayer *conv1 = network->addConvolutionNd(input, outFeatureMap, nvinfer1::DimsHW{kernelSize, kernelSize}, weightMap[layerName + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(nvinfer1::DimsHW{stride, stride});
    conv1->setPaddingNd(nvinfer1::DimsHW{padding, padding});
    conv1->setNbGroups(numGroups);
    nvinfer1::IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), layerName + ".bn", 1e-3);

    // hard_swish = x * hard_sigmoid
    auto hsig = network->addActivation(*bn1->getOutput(0), nvinfer1::ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

nvinfer1::ILayer *FaceDetectionYolov5::focus(nvinfer1::INetworkDefinition *network,
                                             std::map<std::string, nvinfer1::Weights> &weightMap,
                                             nvinfer1::ITensor &input,
                                             int inputChannels, int outputChannels,
                                             int kernelSize, std::string layerName)
{
    nvinfer1::ISliceLayer *s1 = network->addSlice(input, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{inputChannels, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, nvinfer1::Dims3{1, 2, 2});
    nvinfer1::ISliceLayer *s2 = network->addSlice(input, nvinfer1::Dims3{0, 1, 0}, nvinfer1::Dims3{inputChannels, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, nvinfer1::Dims3{1, 2, 2});
    nvinfer1::ISliceLayer *s3 = network->addSlice(input, nvinfer1::Dims3{0, 0, 1}, nvinfer1::Dims3{inputChannels, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, nvinfer1::Dims3{1, 2, 2});
    nvinfer1::ISliceLayer *s4 = network->addSlice(input, nvinfer1::Dims3{0, 1, 1}, nvinfer1::Dims3{inputChannels, Yolo::INPUT_H / 2, Yolo::INPUT_W / 2}, nvinfer1::Dims3{1, 2, 2});

    nvinfer1::ITensor *inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outputChannels, kernelSize, 1, 1, layerName + ".conv");
    return conv;
}

nvinfer1::ILayer *FaceDetectionYolov5::bottleneck(nvinfer1::INetworkDefinition *network,
                                                  std::map<std::string, nvinfer1::Weights> &weightMap,
                                                  nvinfer1::ITensor &input, int c1, int c2,
                                                  bool shortcut, int g, float e, std::string layerName)
{
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 1, 1, 1, layerName + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, layerName + ".cv2");
    if (shortcut && c1 == c2)
    {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

nvinfer1::ILayer *FaceDetectionYolov5::bottleneckCSP(nvinfer1::INetworkDefinition *network,
                                                     std::map<std::string, nvinfer1::Weights> &weightMap,
                                                     nvinfer1::ITensor &input, int c1, int c2,
                                                     int n, bool shortcut, int g, float e, std::string layerName)
{
    nvinfer1::Weights emptywts{nvinfer1::DataType::kFLOAT, nullptr, 0};
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, layerName + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, nvinfer1::DimsHW{1, 1}, weightMap[layerName + ".cv2.weight"], emptywts);
    nvinfer1::ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++)
    {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, layerName + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, nvinfer1::DimsHW{1, 1}, weightMap[layerName + ".cv3.weight"], emptywts);

    nvinfer1::ITensor *inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    nvinfer1::IScaleLayer *bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), layerName + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), nvinfer1::ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, layerName + ".cv4");
    return cv4;
}

nvinfer1::ILayer *FaceDetectionYolov5::SPP(nvinfer1::INetworkDefinition *network,
                                           std::map<std::string, nvinfer1::Weights> &weightMap,
                                           nvinfer1::ITensor &input, int c1, int c2,
                                           int k1, int k2, int k3, std::string layerName)
{
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, layerName + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k1, k1});
    pool1->setPaddingNd(nvinfer1::DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(nvinfer1::DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k2, k2});
    pool2->setPaddingNd(nvinfer1::DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(nvinfer1::DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{k3, k3});
    pool3->setPaddingNd(nvinfer1::DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(nvinfer1::DimsHW{1, 1});

    nvinfer1::ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, layerName + ".cv2");
    return cv2;
}

nvinfer1::IDeconvolutionLayer *FaceDetectionYolov5::deconvBlock(nvinfer1::INetworkDefinition *network,
                                                                std::map<std::string, nvinfer1::Weights> &weightMap,
                                                                nvinfer1::ITensor &input,
                                                                int k, std::string layerName)
{
    nvinfer1::Weights emptyWts = {nvinfer1::DataType::kFLOAT, nullptr, 0};

    float *deconvVal = reinterpret_cast<float *>(malloc(sizeof(float) * k * 2 * 2));
    for (int i = 0; i < k; ++i)
        deconvVal[i] = 1.0;

    nvinfer1::Weights deconvWts = {nvinfer1::DataType::kFLOAT, deconvVal, k * 2 * 2};
    nvinfer1::IDeconvolutionLayer *deconv = network->addDeconvolutionNd(input, k, nvinfer1::DimsHW{2, 2}, deconvWts, emptyWts);
    deconv->setStrideNd(nvinfer1::DimsHW{2, 2});
    deconv->setNbGroups(k);
    weightMap[layerName] = deconvWts;

    return deconv;
}