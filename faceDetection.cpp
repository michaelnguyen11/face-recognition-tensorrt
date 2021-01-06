#include "faceDetection.h"

FaceDetection::FaceDetection()
{
    m_faceThreshold = 0.8;
    m_nmsThreshold = 0.5;

    m_interpreter = std::make_unique<tflite::Interpreter>();

    std::string modelPath = "data/face_ssd_mobilenet_v2.tflite";
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
}

FaceDetection::~FaceDetection() {}

void FaceDetection::preprocess(const cv::Mat &frame)
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

void FaceDetection::postprocess(const cv::Mat &frame,
                                std::vector<cv::Rect> &boxes,
                                std::vector<float> &scores)
{
    TfLiteTensor *detectionBoxes = m_interpreter->tensor(m_interpreter->outputs()[0]);
    TfLiteTensor *detectionClasses = m_interpreter->tensor(m_interpreter->outputs()[1]);
    TfLiteTensor *detectionScores = m_interpreter->tensor(m_interpreter->outputs()[2]);
    TfLiteTensor *numberBoxes = m_interpreter->tensor(m_interpreter->outputs()[3]);

    const float *detectionBoxesData = detectionBoxes->data.f;
    const float *detectionClassesData = detectionClasses->data.f;
    const float *detectionScoresData = detectionScores->data.f;
    const float *numberBoxesData = numberBoxes->data.f;

    std::vector<float> locations, cls;
    int numOfDetectionAllowed = 20;
    for (int i = 0; i < numOfDetectionAllowed; ++i)
    {
        locations.push_back(detectionBoxesData[i]);
        cls.push_back(detectionClassesData[i]);
    }

    std::vector<cv::Rect> outputBoxes;
    std::vector<float> outputScores;

    for (int i = 0; i < *numberBoxesData; i++)
    {
        int ymin = locations[i * 4] * frame.rows;
        int xmin = locations[i * 4 + 1] * frame.cols;
        int ymax = locations[i * 4 + 2] * frame.rows;
        int xmax = locations[i * 4 + 3] * frame.cols;

        outputScores.push_back(detectionScoresData[i]);
        outputBoxes.push_back(cv::Rect(xmin, ymin, (xmax - xmin), (ymax - ymin)));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(outputBoxes, outputScores, m_faceThreshold, m_nmsThreshold, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        boxes.push_back(outputBoxes[i]);
        scores.push_back(outputScores[i]);
    }
}

void FaceDetection::detectFace(const cv::Mat &frame,
                               std::vector<cv::Rect> &boxes,
                               std::vector<float> &scores)
{
    preprocess(frame);
    m_interpreter->Invoke();
    postprocess(frame, boxes, scores);
}

void FaceDetection::detectSingleFace(const cv::Mat &frame, cv::Rect &bbox)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    preprocess(frame);
    m_interpreter->Invoke();
    postprocess(frame, boxes, scores);

    bbox = boxes[0];
}