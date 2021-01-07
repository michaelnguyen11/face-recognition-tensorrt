#include <iostream>
#include <opencv2/opencv.hpp>

#include "cppSocket.h"

int main()
{
    cv::VideoCapture cap(2);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        return 0;
    }
    cap.set(3, 1280);
    cap.set(4, 720);
    cv::Mat frame, frameResult;

    CppSocket client = CppSocket(false);

    while(cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        int send = client.sendImage(frame);
        if (send < 0)
        {
            std::cerr << "Failed to send image" << std::endl;
            continue;
        }

        int recv = client.receiveImage(frameResult);
        if (recv < 0)
        {
            std::cerr << "Failed to receive image" << std::endl;
            continue;
        }

        cv::imshow("result", frameResult);
        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(30);
        if (c == 27)
            break;
    }
    client.disconnect();

    return 0;
}
