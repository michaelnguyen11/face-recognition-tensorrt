#include <iostream>
#include <memory>
#include <chrono>
#include <numeric>
#include <sys/stat.h>
#include <future>

#include <opencv2/opencv.hpp>
#include "faceDetection.h"
#include "faceRecognition.h"
#include "faceAntiSpoofing.h"

const int g_numClass = 192;
std::unordered_map<std::string, std::pair<cv::Mat, std::array<float, g_numClass>>> g_faceDatabase;

void initDatabase(const std::string &dataDir,
                  const std::unique_ptr<FaceDetection> &fd,
                  const std::unique_ptr<MobileFaceNet> &mfn)
{
    std::vector<cv::String> imageNames;
    const int dirError = mkdir(dataDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (dirError != -1 && errno != EEXIST)
    {
        return;
    }

    cv::glob(dataDir, imageNames, true);
    for (size_t i = 0; i < imageNames.size(); ++i)
    {
        cv::Mat image = cv::imread(imageNames[i]);
        cv::Rect bbox;
        fd->detectSingleFace(image, bbox);
        if (bbox.empty())
        {
            std::cout << "Face is not found" << std::endl;
            continue;
        }
        cv::Mat face = image(bbox);
        std::array<float, g_numClass> featureVec = mfn->extractFeatures(face);
        std::string label = imageNames[i];
        label.erase(0, dataDir.length());
        label.erase(label.find("."));
        g_faceDatabase[label] = {face, featureVec};
    }
}

std::pair<std::string, float> predictLabel(std::array<float, g_numClass> &predict,
                                           float threshold = 0.5)
{
    float ret = 0, mod1 = 0, mod2 = 0;
    float max = 0;

    std::string label;
    for (int i = 0; i < g_numClass; ++i)
    {
        mod1 += predict[i] * predict[i];
    }

    for (auto &face : g_faceDatabase)
    {
        std::array<float, g_numClass> &vec = face.second.second;
        for (int i = 0; i < g_numClass; ++i)
        {
            ret += predict[i] * vec[i];
            mod2 += vec[i] * vec[i];
        }
        float cosineSimilarity = (ret / sqrt(mod1) / sqrt(mod2) + 1) / 2.0;
        if (max < cosineSimilarity)
        {
            max = cosineSimilarity;
            label = face.first;
        }
    }
    if (max < threshold)
        label.clear();

    return {label, max};
}

int main(int argc, char **argv)
{
    std::unique_ptr<FaceDetection> fd = std::make_unique<FaceDetection>();
    std::unique_ptr<MobileFaceNet> mfn = std::make_unique<MobileFaceNet>();
    std::unique_ptr<FaceAntiSpoofing> fas = std::make_unique<FaceAntiSpoofing>();

    std::string dataDir = "face_database/";
    initDatabase(dataDir, fd, mfn);

    int frameCount = 0;
    std::vector<int> programTime;
    float antiSpoofingThreshold = 0.89;

    cv::VideoCapture cap(2);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return 0;
    }
    cap.set(3, 1280);
    cap.set(4, 720);

    cv::Mat frame;

    while (cap.isOpened())
    {
        auto start = std::chrono::system_clock::now();

        cap >> frame;
        frameCount++;
        if (frame.empty())
        {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        fd->detectFace(frame, bboxes, scores);

        for (size_t i = 0; i < bboxes.size(); ++i)
        {
            cv::Rect bbox = bboxes[i];
            bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);

            // Face Anti Spoofing
            float antiSpoofConf = fas->detect(frame, bbox);
            if (antiSpoofConf > antiSpoofingThreshold)
            {
                // Face Recognition
                cv::Mat face = frame(bbox);
                std::array<float, g_numClass> features = mfn->extractFeatures(face);
                std::pair<std::string, float> resultFeatures = predictLabel(features);
                std::string labelOut = resultFeatures.first;
                if (!labelOut.empty())
                {
                    cv::putText(frame, labelOut, cv::Point(bbox.x, bbox.y * 0.8), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 255, 30));
                    cv::putText(frame, "similarity: " + std::to_string(resultFeatures.second), cv::Point(bbox.x, bbox.y * 0.7), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(255, 255, 0));
                }
                cv::putText(frame, "TRUE Face", cv::Point(bbox.width + bbox.x, bbox.height + bbox.y), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 0), 2);
            }
            else
                cv::putText(frame, "FAKE Face", cv::Point(bbox.width + bbox.x, bbox.height + bbox.y), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 0), 2);

            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 1);
            cv::putText(frame, std::to_string(scores[i]), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
        }

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        programTime.push_back(elapsed);

        if (frameCount % 100 == 0)
        {
            int averageMs = std::accumulate(programTime.begin(), programTime.end(), 0) / programTime.size();
            std::cout << "Average computation time: " << averageMs << " ms" << std::endl;
            programTime.clear();
        }

        cv::imshow("test", frame);
        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(30);
        if (c == 27)
            break;
    }

    return 0;
}
