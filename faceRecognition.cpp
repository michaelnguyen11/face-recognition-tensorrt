#include "faceRecognition.h"

MobileFaceNet::MobileFaceNet()
{
    std::string modelPath = "data/mobileFaceNet.tflite";
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());

    if (!m_model)
    {
        std::cerr << "Failed to mmap tflite model" << std::endl;
        exit(0);
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*m_model.get(), resolver)(&m_interpreter) != kTfLiteOk)
    {
        std::cerr << "Failed to interpreter tflite model" << std::endl;
        exit(0);
    }

#ifdef GPU_DELEGATE
    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, // FP16
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };

    TfLiteDelegate *delegate = TfLiteGpuDelegateV2Create(&options);
    if (m_interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
    {
        std::cerr << "Failed to modify graph with delegate" << std::endl;
        exit(0);
    }
#endif

    // feed input
    if (m_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Failed to allocate tensors" << std::endl;
        exit(0);
    }

    // Find input tensors.
    if (m_interpreter->inputs().size() != 1)
    {
        std::cerr << "Graph needs to have 1 and only 1 input!" << std::endl;
        exit(0);
    }
    m_interpreter->SetAllowFp16PrecisionForFp32(true);
    m_interpreter->SetNumThreads(4);
}

MobileFaceNet::~MobileFaceNet()
{
}

void MobileFaceNet::preprocess(const cv::Mat &frame)
{
    TfLiteTensor *inputTensor = m_interpreter->tensor(m_interpreter->inputs()[0]);

    cv::Mat resizedFrame = frame.clone();
    // Expected model dims
    TfLiteIntArray *dims = inputTensor->dims;
    int expectedHeight = dims->data[1];
    int expectedWidth = dims->data[2];
    int expectedChannels = dims->data[3];

    // preprocess input frame as expected model dims
    cv::resize(frame, resizedFrame, cv::Size(expectedWidth, expectedHeight));
    if (inputTensor->type == kTfLiteFloat32)
    {
        // Normalize the image based on std and mean (p' = (p-mean)/std)
        resizedFrame.convertTo(resizedFrame, CV_32FC3, 1 / 128.0, -128.0 / 128.0);

        // Copy image into input tensor
        float *dst = inputTensor->data.f;
        memcpy(dst, resizedFrame.data, sizeof(float) * expectedWidth * expectedHeight * expectedChannels);
    }
    else if (inputTensor->type == kTfLiteUInt8)
    {
        // Copy image into input tensor
        uchar *dst = inputTensor->data.uint8;
        memcpy(dst, resizedFrame.data, sizeof(uchar) * expectedWidth * expectedHeight * expectedChannels);
    }
    else
    {
        std::cerr << "The input tensor type " << inputTensor->type << " has not supported yet." << std::endl;
        return;
    }
}

std::array<float, g_mfnNumClass> MobileFaceNet::extractFeatures(const cv::Mat &frame)
{
    std::array<float, g_mfnNumClass> embeddingsArray;
    this->preprocess(frame);

    m_interpreter->Invoke();

    TfLiteTensor *embeddingsTensor = m_interpreter->tensor(m_interpreter->outputs()[0]);
    TfLiteIntArray *embeddingsDims = embeddingsTensor->dims;
    auto embeddingsSize = embeddingsDims->data[embeddingsDims->size - 1];
    float *embeddingsData = embeddingsTensor->data.f;
    for (int i = 0; i < embeddingsSize; ++i)
    {
        embeddingsArray[i] = embeddingsData[i];
    }

    if (embeddingsTensor->type == kTfLiteFloat32)
    {
        float *embeddingsData = embeddingsTensor->data.f;
        for (int i = 0; i < embeddingsSize; ++i)
            embeddingsArray[i] = embeddingsData[i];
    }
    else if (embeddingsTensor->type == kTfLiteUInt8)
    {
        uint8_t *embeddingsData = embeddingsTensor->data.uint8;
        for (int i = 0; i < embeddingsSize; ++i)
            embeddingsArray[i] = embeddingsData[i];
    }
    else
    {
        std::cerr << "The output tensor type " << embeddingsTensor->type << " has not supported yet." << std::endl;
    }

    return embeddingsArray;
}
