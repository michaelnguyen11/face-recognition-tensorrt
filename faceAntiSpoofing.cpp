#include "faceAntiSpoofing.h"

FaceAntiSpoofing::FaceAntiSpoofing()
{
    std::ifstream configFile("data/face_anti_spoofing_config.json");
    Json::Reader reader;
    reader.parse(configFile, m_jsonObject);

    m_inputName = "data";
    m_outputName = "softmax";

    m_ncnnOption.lightmode = true;
    m_ncnnOption.num_threads = 2;

    this->loadModel();
}

FaceAntiSpoofing::~FaceAntiSpoofing()
{
}

void FaceAntiSpoofing::loadModel()
{
    const Json::Value &faceAntiSpoofingConfig = m_jsonObject["face_anti_spoofing"];
    for (int i = 0; i < faceAntiSpoofingConfig.size(); ++i)
    {
        std::string param = faceAntiSpoofingConfig[i]["model_path"].asString() + ".param";
        std::string model = faceAntiSpoofingConfig[i]["model_path"].asString() + ".bin";

        std::shared_ptr<ncnn::Net> net = std::make_shared<ncnn::Net>();
        net->opt = m_ncnnOption;
        net->load_param(param.c_str());
        net->load_model(model.c_str());
        m_ncnnNetworks.emplace_back(net);
    }
}

float FaceAntiSpoofing::detect(cv::Mat &frame, const cv::Rect &bbox)
{
    const Json::Value &faceAntiSpoofingConfig = m_jsonObject["face_anti_spoofing"];
    float confidence = 0.f; //score

    for (int i = 0; i < faceAntiSpoofingConfig.size(); i++)
    {
        cv::Mat roi;
        if (faceAntiSpoofingConfig[i]["original_size"].asBool())
        {
            cv::resize(frame, roi, cv::Size(80, 80), 0, 0, 3);
        }
        else
        {
            cv::Rect rect = this->calculateBox(bbox, frame.cols, frame.rows, faceAntiSpoofingConfig[i]);
            cv::resize(frame(rect), roi, cv::Size(faceAntiSpoofingConfig[i]["width"].asInt(), faceAntiSpoofingConfig[i]["height"].asInt()));
        }

        ncnn::Mat input = ncnn::Mat::from_pixels(roi.data, ncnn::Mat::PIXEL_BGR, roi.cols, roi.rows);

        ncnn::Extractor extractor = m_ncnnNetworks[i]->create_extractor();
        extractor.set_light_mode(true);
        extractor.set_num_threads(2);

        extractor.input(m_inputName.c_str(), input);
        ncnn::Mat output;
        extractor.extract(m_outputName.c_str(), output);

        confidence += output.row(0)[1];
    }

    confidence /= faceAntiSpoofingConfig.size();

    return confidence;
}

cv::Rect FaceAntiSpoofing::calculateBox(const cv::Rect &faceBbox, int width, int height, const Json::Value &config)
{
    int x = static_cast<int>(faceBbox.x);
    int y = static_cast<int>(faceBbox.y);
    int bboxWidth = static_cast<int>(faceBbox.width + 1);
    int bboxHeight = static_cast<int>(faceBbox.height + 1);

    int shift_x = static_cast<int>(bboxWidth * config["shift_x"].asFloat());
    int shift_y = static_cast<int>(bboxHeight * config["shift_y"].asFloat());

    float scale = std::min(
        config["scale"].asFloat(),
        std::min((width - 1) / (float)bboxWidth, (height - 1) / (float)bboxHeight));

    int bboxCenterX = bboxWidth / 2 + x;
    int bboxCenterY = bboxHeight / 2 + y;

    int newWidth = static_cast<int>(bboxWidth * scale);
    int newHeight = static_cast<int>(bboxHeight * scale);

    int leftTopX = bboxCenterX - newWidth / 2 + shift_x;
    int leftTopY = bboxCenterY - newHeight / 2 + shift_y;
    int rightBottomX = bboxCenterX + newWidth / 2 + shift_x;
    int rightBottomY = bboxCenterY + newHeight / 2 + shift_y;

    if (leftTopX < 0)
    {
        rightBottomX -= leftTopX;
        leftTopX = 0;
    }

    if (leftTopY < 0)
    {
        rightBottomY -= leftTopY;
        leftTopY = 0;
    }

    if (rightBottomX >= width)
    {
        int s = rightBottomX - width + 1;
        leftTopX -= s;
        rightBottomX -= s;
    }

    if (rightBottomY >= height)
    {
        int s = rightBottomY - height + 1;
        leftTopY -= s;
        rightBottomY -= s;
    }

    return cv::Rect(leftTopX, leftTopY, newWidth, newHeight);
}