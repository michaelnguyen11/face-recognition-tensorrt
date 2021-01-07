#include "cppSocket.h"

CppSocket::CppSocket(bool isModeServer, std::string serverAddress, int port)
    : m_serverAddress(serverAddress), m_port(port)
{
    // Retry to connect 5 time when server is busy
    m_reconnectOnAddressBusy = 5;

    // Setup socket for client and server
    m_servAddr.sin_family = AF_INET;
    m_servAddr.sin_port = htons(m_port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, m_serverAddress.c_str(), &m_servAddr.sin_addr) <= 0)
        std::cout << "Invalid address. Address not supported" << std::endl;

    std::cout << "Starting " << (isModeServer ? "Server" : "Client") << std::endl;
    std::cout << "Starting up on " << m_serverAddress << ", port " << m_port << std::endl;

    bool addressFreeFlag = false;
    if (isModeServer)
    {
        int opt = 1;
        int addrlen = sizeof(m_servAddr);
        // Creating socket file descriptor
        if ((m_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        {
            std::cerr << "Socket creation failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        else
            std::cout << "Socket creation successful" << std::endl;

        // Forcefully attaching socket to the port
        if (setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
                       &opt, sizeof(opt)))
        {
            std::cerr << "Setsockopt failed" << std::endl;
            exit(EXIT_FAILURE);
        }
        m_servAddr.sin_addr.s_addr = INADDR_ANY;

        while (!addressFreeFlag)
        {
            // Forcefully attaching socket to the specified port
            if (bind(m_socket, (struct sockaddr *)&m_servAddr,
                     sizeof(m_servAddr)) < 0)
            {
                printf("\nServer Binding to Address Failed...\n");
                if (m_reconnectOnAddressBusy == 0)
                    std::cout << "Please make sure address is free, else use m_reconnectOnAddressBusy argument "
                              << "to keep polling in periodic intervals. If you have just run a server previously "
                              << "there's a good chance the previous server will be down in a couple of seconds."
                              << "Use polling functionality to avoid waiting for address to be free again.";
                pollingTimeout();
            }
            else
                addressFreeFlag = true;
        }

        if (listen(m_socket, 5) < 0) // queue of pending connections -> 1
        {
            std::cerr << "listen" << std::endl;
            exit(EXIT_FAILURE);
        }

        this->waitForConnection();
    }
    else
    {
        if ((m_sockConn = socket(AF_INET, SOCK_STREAM, 0)) < 0)
            printf("Socket creation error\n");
        else
            printf("Socket creation successful\n");

        while (!addressFreeFlag)
        {
            std::cout << "Client is waiting to connect to server...\n";
            if (connect(m_sockConn, (struct sockaddr *)&m_servAddr, sizeof(m_servAddr)) < 0)
            {
                std::cerr << "Client Connection to Server Failed" << std::endl;
                this->pollingTimeout();
            }
            else
            {
                std::cout << "Client Socket connection to Server Successful" << std::endl;
                addressFreeFlag = true;
            }
        }
    }
}

CppSocket::~CppSocket()
{
    disconnect();
}

void CppSocket::waitForConnection()
{
    int addrlen = sizeof(m_servAddr);
    printf("Waiting for a connection ...\n");
    if ((m_sockConn = accept(m_socket, (struct sockaddr *)&m_servAddr,
                            (socklen_t *)&addrlen)) < 0)
    {
        std::cerr << "Failed to accept connection from client" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Connected IP address: " << inet_ntoa(m_servAddr.sin_addr) << ":" << htons(m_servAddr.sin_port) << std::endl;
    std::cout << "Connection established ..." << std::endl;
}

void CppSocket::pollingTimeout()
{
    if (m_reconnectOnAddressBusy > 0)
    {
        std::cout << "Will attempt to reconnect in " << m_reconnectOnAddressBusy << " seconds" << std::endl;
        usleep((uint32_t)(m_reconnectOnAddressBusy * 1000000));
    }
    else
    {
        if (m_reconnectOnAddressBusy != 0)
            std::cout << "Invalid value passed to reconnectOnAddressBusy argument" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CppSocket::disconnect()
{
    shutdown(m_sockConn, SHUT_RDWR);
    close(m_sockConn);

    this->waitForConnection();
}

int CppSocket::receiveImage(cv::Mat &image)
{
    int headerBuffers[HDR_BYTES/4];
    memset(&headerBuffers, 0, sizeof(headerBuffers));

    recv(m_sockConn, (char *)(&headerBuffers), HDR_BYTES, 0);

    int imageWidth = headerBuffers[1];
    int imageHeight = headerBuffers[2];
    int imageChannels = headerBuffers[3];

    if (imageWidth == 0)
    {
        return -1;
    }

    cv::Mat img;
    if (imageChannels == 1)
        img = cv::Mat(cv::Size(imageWidth, imageHeight), CV_8UC1);
    else
        img = cv::Mat(cv::Size(imageWidth, imageHeight), CV_8UC3);

    int imageSize = headerBuffers[0] - HDR_BYTES;
    char *receivedBuffer = new char[imageSize];

    int pos = 0;
    int length = 0;
    while (pos < imageSize)
    {
        length = recv(m_sockConn, (char *)receivedBuffer + pos + HDR_BYTES/4, imageSize - pos, 0);
        if (length < 0)
        {
            printf("Server Recieve Data Failed!\n");
            return -1;
            break;
        }
        pos += length;
    }

    memcpy(img.data, receivedBuffer + HDR_BYTES, imageSize);

    image = img;

    delete[] receivedBuffer;

    return 1;
}

int CppSocket::sendImage(cv::Mat &image)
{
    if (image.empty())
    {
        printf("empty image\n\n");
        return -1;
    }

    int imageSize = image.rows * image.cols * image.channels();
    int packageSize = imageSize + HDR_BYTES;

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    int headerBuffers[HDR_BYTES];

    headerBuffers[0] = packageSize;
    headerBuffers[1] = width;
    headerBuffers[2] = height;
    headerBuffers[3] = channels;

    char *sendBuffers = new char[packageSize];

    for (int i = 0; i < HDR_BYTES/4; i++)
    {
        *(((int *)sendBuffers) + i) = headerBuffers[i];
    }

    memcpy(sendBuffers + HDR_BYTES, image.data, imageSize);

    if (send(m_sockConn, (char *)sendBuffers, packageSize, 0) < 0)
    {
        printf("send image error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    delete[] sendBuffers;

    return 1;
}

bool CppSocket::isClientConnected()
{
    char buf;
    // Check receive 1 byte from client, the MSG_PEEK mean data is still readable
    int err = recv(m_sockConn, &buf, 1, MSG_PEEK);
    if (err == SO_ERROR)
    {
        if (errno != EWOULDBLOCK)
            return false;
    }

    return true;
}
