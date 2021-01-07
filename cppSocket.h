#ifndef CPP_SOCKET_H_
#define CPP_SOCKET_H_

#include <iostream>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <cstring>

#include <opencv2/opencv.hpp>

#define HDR_BYTES 16

class CppSocket
{
public:
    CppSocket(bool isModeServer, std::string serverAddress = "127.0.0.1", int port = 2222);
    ~CppSocket();

    int sendImage(cv::Mat &image);
    int receiveImage(cv::Mat &image);
    void disconnect();
    void waitForConnection();
    bool isClientConnected();

private:
    void pollingTimeout();
    void packageImage(char *imageBuffer, cv::Mat &image);

    int m_socket;
    int m_sockConn;
    int m_reconnectOnAddressBusy;
    std::string m_serverAddress;
    int m_port;
    struct sockaddr_in m_servAddr;
};

#endif // CPP_SOCKET_H_
