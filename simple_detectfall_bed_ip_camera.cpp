//============================================================================
// Name        : fall_bed_camera.cpp
// Author      : hugoliu
// Version     :
// Copyright   : copyright to hugoliu
// Description : Hello World in C++, Ansi-style
//============================================================================
// Design by hugo liu
// Date 20210112
// fall detection & bed detection for TI tof camera
// modify opencv4.5
//#define Wayne_debug //2022
//#pragma warning(disable:4996) //2022 prevent sprinf error pop
#include <stdio.h>
#include <string.h>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <vector>
#include <locale>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <codecvt>
//V4L

#include <sys/ioctl.h>
#include <getopt.h>             //getopt_long() 
#include <fcntl.h>              // low-level i/o 
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>
//network
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <net/if.h>   //ifreq
#include <sys/ioctl.h>

//opencv
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/types_c.h"

//#define net_udp
//#define UDP_SERVER_PORT 5001
#define IP_CAMERA
#define TCP_SERVER_PORT 23311
//#define MB 640*480
//#define  DEBUG_PRINT
#define CAMERA_MODE
#define SAVE_DATA

//#define VIDEO_RECORD
//#define RESULT_VIDEO

//#define SHOW_MULTI_PEOPLE
#define NO_REPEAT

//#define Compare_LOG
//#define HUGO_DEBUG
//#define IMAGE_MASK

#define EDGE_LEVEL 10

#ifdef CAMERA_MODE
#define CLEAR(x) memset(&(x), 0, sizeof(x))
#endif
#define TOF_DEPTH_WIDTH       80
#define TOF_DEPTH_HEIGHT      60
#define TOF_DEPTH_PIXELS      (TOF_DEPTH_WIDTH * TOF_DEPTH_HEIGHT)
#define TOF_DEPTH_ONLY_FRAME_SIZE    (TOF_DEPTH_WIDTH * TOF_DEPTH_HEIGHT * sizeof(unsigned short))
//#define TOF_IR_ONLY_FRAME_SIZE    TOF_DEPTH_ONLY_FRAME_SIZE
#define TOF_DEPTH_IR_FRAME_SIZE (TOF_DEPTH_ONLY_FRAME_SIZE * 2)


#define PRE_FALL_VALUE 10
//#define IMAGE_PATH "D:\\work\\Tof\\TI_work\\data\\icon\\"

#define WORK_PATH   "/home/hugoliu/work/fall_bed/"
#define IMAGE_PATH "/home/hugoliu/work/fall_bed/icon/"
#define NO_FRAME_NO

#define high_diff_th           160
#define LINE_AVG_HIGH_240       29  
#define LINE_AVG_HIGH_250       30
#define LINE_AVG_HIGH_270       32

//#define UP_DIFF_THRESHOLD_240    5
//#define UP_DIFF_THRESHOLD_250   10
#define UP_HIGH_LEVEL_240       21 //22
#define UP_HIGH_LEVEL_250       30 //28

#define DOWN_LEVEL_TH_240       13
#define DOWN_LEVEL_TH_250       15

#define WIDTH_TH     25
#define BED_LENGTH   50

//#define IMAGE_INIT
//#define IMAGE_MASK
//#define HTTP_ENABLE
//#define HTTP_ENABLE_RESP
//#define PORT_NO 7001
//#define MAX_DEPTH_LOC
//#define DEBUG_X
//#define DEBUG_X_FIG
//#define DEBUG_Y
//#define DEBUG_NEW
//#define ENGINEER
//#define display_message  //bed detect
//#define show_message     //fall
//#define show_message_pre     //fall
//#define DoNotDisplayRightLeft
#define SHOW_BED_TOF
//#define LOG_FILE
//#define PEOPLE_HIGH
//#define PEOPLE_HAVE

//#define DEBUG_FALL_SLOW
//#define BED_LINE_DEBUG
//#define BED_COL_DEBUG
//#define DEBUG_PEOPLE_COL
//#define DEBUG_PEOPLE_ROW
//#define SAVE_JPG

#ifdef SAVE_JPG
#define BED_LINE_DEBUG_FILE
#define BED_COL_DEBUG_FILE
#define IMAGE_INIT
#endif

using namespace std;
using namespace cv;

const static int P_height = 50; //50
const static int P_area = 6; // 2000; // 2000; // 9000;
//const static int P_area2 = 48; // 3800; // 8000; // 9000;

//bool ldown = false, lup = false;
int fontFace = FONT_HERSHEY_COMPLEX; // FONT_HERSHEY_SIMPLEX;
double fontScale = 0.8;
int thickness = 2;
char text[30];
int tmpNowMin;
int max_hight_now = 0;
Mat tof_depth = Mat(TOF_DEPTH_HEIGHT, TOF_DEPTH_WIDTH, CV_16UC1, Scalar(0));
Mat tof_ir = Mat(TOF_DEPTH_HEIGHT, TOF_DEPTH_WIDTH, CV_16UC1, Scalar(0));

Mat image = Mat(60, 80, CV_8UC1, Scalar(0));
Mat image_ir = Mat(60, 80, CV_8UC1, Scalar(0));
Mat resize_img = Mat(240, 320, CV_8UC1, Scalar(0));

struct Bed {
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    int width_x;
    int width_y;
    int start2_y;
    int end2_y;
    int width2_y;
    int center_x;
    int center_y;
    int center_high;
    int before_y;
    int before_x;
    int before_high;

    int area;
    int area_max;
    int area_low;
    int area_middle;
    int area_high;
    int area_half_max;
    int area_half;
    int area_half_high;
    int width_div_2;
    int hight_div_2;
};
int fall_area_start_x = 0, fall_area_end_x = 0;
int fall_area_start_y = 0, fall_area_end_y = 0;
int max_high_fall_area = 0;
int area_high_x = 0;
int area_high_y = 0;
int center_area_start_x = 0;
int center_area_start_y = 0;
int center_area_end_x = 0;
int center_area_end_y = 0;


/*
#ifdef IP_CAMERA
static const int kTextSize = 10;

int sendall(int socket, char * buf, int *len){
    int total = 0;        // how many bytes we've sent
    int bytesleft = *len; // how many we have left to send
    int n;

    while(total < *len) {
        n = send(socket, buf + total, bytesleft, 0);
        if (n == -1) { break; }
        total += n;
        bytesleft -= n;
    }

    *len = total; // return number actually sent here

    return n==-1?-1:0; // return -1 on failure, 0 on success
}
UTF-8
std::wstring s2ws(const std::string& str) {
  if (str.empty()) {
    return L"";
  }
  unsigned len = str.size() + 1;
  setlocale(LC_CTYPE, "");
  wchar_t *p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring w_str(p);
  delete[] p;
  return w_str;
}

std::string ws2s(const std::wstring& w_str) {
    if (w_str.empty()) {
      return "";
    }
    unsigned len = w_str.size() * 4 + 1;
    setlocale(LC_CTYPE, "");
    char *p = new char[len];
    wcstombs(p, w_str.c_str(), len);
    std::string str(p);
    delete[] p;
    return str;
}

#endif
*/
#ifndef CAMERA_MODE

static void mouse_callback(int event, int x, int y, int, void*)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        //image, image_ir
        Mat local_img;
        /*
        Mat resize_img_ir, Cresize_img_ir;
        resize(image, resize_img, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
        resize(image_ir, resize_img_ir, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
        Mat show_image_all = Mat(480,320, CV_8UC3, Scalar(0));
        cvtColor(resize_img_ir, Cresize_img_ir, CV_GRAY2RGB);
       */
        cvtColor(resize_img, local_img, CV_GRAY2RGB);
        int local_depth = resize_img.at<uchar>(y, x);
        //		sprintf(text, "(%d, %d, %d)", x, y, local_depth);
        int dist_y, dist_x;
        if (x > 220) dist_x = 220;
        else if (x < 5) dist_x = 5;
        else dist_x = x;
        if (y > 190) dist_y = 190;
        else if (y < 5) dist_y = 5;
        else dist_y = y;
        int real_x = x / 4;
        int real_y = y / 4;
        int people_high_temp = ((max_hight_now - local_depth) << 4) / 10;
        sprintf(text, "(%d,%d)", real_x, real_y);
        putText(local_img, text, Point(dist_x, dist_y), fontFace, fontScale, Scalar(240, 129120), thickness, 1);
        sprintf(text, "(%d,%d)", local_depth, people_high_temp);
        putText(local_img, text, Point(dist_x, dist_y + 40), fontFace, fontScale, Scalar::all(255), thickness, 1);
        /*
        local_img.copyTo(show_image_all(Rect(0, 0, 320, 240)));
        Cresize_img_ir.copyTo(show_image_all(Rect(0, 240, 320, 240)));
        imshow("stopshow", show_image_all);
        */
        imshow("stopshow", local_img);
        // resize_img
    }
}
#endif


#ifdef CAMERA_MODE
#define CLEAR(x) memset(&(x), 0, sizeof(x))
char cmd_tof_resp[128];

struct buffer {
    void* start;
    size_t  length;
};

static char* uvc_dev_name;
static char* tty_dev_name;
int             fd_uvc = -1;
static int              fd_tty = -1;
struct buffer* buffers;
unsigned int     n_buffers;


static void errno_exit(const char* s)
{
    printf("RebootX\n");
    printf("%s error %d, %s\n", s, errno, strerror(errno));
    //exit(EXIT_FAILURE);
    system("sync");
    system("sync");
    printf("The reboot \n");
    system("sudo reboot now");
}

static int xioctl(int fh, int request, void* arg)
{
    int r;

    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);

    return r;
}

static int read_frame(void)
{
    struct v4l2_buffer buf;

    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(fd_uvc, &fds);

    /* Timeout. */

    tv.tv_sec = 2;
    tv.tv_usec = 0;

    r = select(fd_uvc + 1, &fds, NULL, NULL, &tv);

    if (-1 == r) {
        if (EINTR != errno) errno_exit("select");
    }

    if (0 == r) {
        printf("select timeout\n");
        exit(EXIT_FAILURE);
    }

    CLEAR(buf);

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd_uvc, VIDIOC_DQBUF, &buf)) {
        switch (errno) {
        case EAGAIN:
            return 0;

        case EIO:

        default:
            errno_exit("VIDIOC_DQBUF");
        }
    }

    assert(buf.index < n_buffers);
    unsigned short* p_frame = (unsigned short*)buffers[buf.index].start;
    //        process_image(buffers[buf.index].start, buf.bytesused);
    memcpy(tof_depth.ptr<unsigned short>(0), p_frame, TOF_DEPTH_ONLY_FRAME_SIZE);
    memcpy(tof_ir.ptr<unsigned short>(0), p_frame + TOF_DEPTH_PIXELS, TOF_DEPTH_ONLY_FRAME_SIZE);
    /*
    printf("depth center = %d, IR center = %d\n",
    tof_depth.at<unsigned short>(TOF_DEPTH_HEIGHT >> 1, TOF_DEPTH_WIDTH >> 1),
    tof_ir.at<unsigned short>(TOF_DEPTH_HEIGHT >> 1, TOF_DEPTH_WIDTH >> 1));
    */


    if (-1 == xioctl(fd_uvc, VIDIOC_QBUF, &buf))
        errno_exit("VIDIOC_QBUF");


    return 1;
}

static void stop_capturing(void)
{
    enum v4l2_buf_type type;


    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd_uvc, VIDIOC_STREAMOFF, &type))
        errno_exit("VIDIOC_STREAMOFF");

}

static void start_capturing(void)
{
    unsigned int i;
    enum v4l2_buf_type type;




    for (i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;

        CLEAR(buf);
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (-1 == xioctl(fd_uvc, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd_uvc, VIDIOC_STREAMON, &type))
        errno_exit("VIDIOC_STREAMON");

}

static void uninit_uvc_device(void)
{
    unsigned int i;


    for (i = 0; i < n_buffers; ++i)
        if (-1 == munmap(buffers[i].start, buffers[i].length))
            errno_exit("munmap");


    free(buffers);
}

static void init_mmap(void)
{
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd_uvc, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            printf("%s does not support "
                "memory mappingn", uvc_dev_name);
            exit(EXIT_FAILURE);
        }
        else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2) {
        printf("Insufficient buffer memory on %s\n",
            uvc_dev_name);
        exit(EXIT_FAILURE);
    }

    buffers = (buffer*)calloc(req.count, sizeof(*buffers));

    if (!buffers) {
        printf("Out of memory\n");
        exit(EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        if (-1 == xioctl(fd_uvc, VIDIOC_QUERYBUF, &buf))
            errno_exit("VIDIOC_QUERYBUF");

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start =
            mmap(NULL /* start anywhere */,
                buf.length,
                PROT_READ | PROT_WRITE /* required */,
                MAP_SHARED /* recommended */,
                fd_uvc, buf.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start)
            errno_exit("mmap");
    }
}

static void init_device(void)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl(fd_uvc, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            printf("%s is no V4L2 device\n",
                uvc_dev_name);
            exit(EXIT_FAILURE);
            //system("sudo reboot");
        }
        else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        printf("%s is no video capture device\n",
            uvc_dev_name);
        exit(EXIT_FAILURE);
        //system("sudo reboot");
    }



    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        printf("%s does not support streaming i/o\n",
            uvc_dev_name);
        exit(EXIT_FAILURE);
        //system("sudo reboot");
    }


    /* Select video input, video standard and tune here. */


    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(fd_uvc, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(fd_uvc, VIDIOC_S_CROP, &crop)) {
            switch (errno) {
            case EINVAL:
                /* Cropping not supported. */
                break;
            default:
                /* Errors ignored. */
                break;
            }
        }
    }
    else {
        /* Errors ignored. */
    }


    CLEAR(fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    /* Note VIDIOC_S_FMT may change width and height. */
    fmt.fmt.pix.width = TOF_DEPTH_WIDTH << 1;
    fmt.fmt.pix.height = TOF_DEPTH_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (-1 == xioctl(fd_uvc, VIDIOC_S_FMT, &fmt))
        // system("sudo reboot");
        errno_exit("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */

/* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;


    init_mmap();

}

static void close_uvc_device(void)
{
    if (-1 == close(fd_uvc))
        errno_exit("close");

    fd_uvc = -1;
}

static void open_uvc_device(void)
{
    struct stat st;

    if (-1 == stat(uvc_dev_name, &st)) {
        printf("Cannot identify '%s': %d, %s\n",
            uvc_dev_name, errno, strerror(errno));
        //system("sudo reboot");
        exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        printf("%s is no devicen", uvc_dev_name);
        //system("sudo reboot");
        exit(EXIT_FAILURE);
    }

    fd_uvc = open(uvc_dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd_uvc) {
        printf("Cannot open '%s': %d, %s\n",
            uvc_dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
        //system("sudo reboot");
    }
}
///////////////////////////tty
static void close_tty_device(void)
{
    if (-1 == close(fd_tty))
        errno_exit("close");

    fd_tty = -1;
}

static void open_tty_device(void)
{
    int status;
    struct stat st;
    struct termios   options;

    if (-1 == stat(tty_dev_name, &st)) {
        fprintf(stderr, "Cannot identify '%s': %d, %s\n",
            tty_dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    fd_tty = open(tty_dev_name, O_RDWR | O_NOCTTY);

    if (fd_tty == -1) {
        fprintf(stderr, "Cannot open '%s': %d, %s\n",
            tty_dev_name, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    status = tcgetattr(fd_tty, &options);
    if (status != 0) {
        fprintf(stderr, "'%s': %d, %s\n",
            tty_dev_name, errno, strerror(errno));
        fprintf(stderr, "tcgetattr error\n");
        exit(EXIT_FAILURE);
    }
    tcflush(fd_tty, TCIOFLUSH);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    /*
     *  Set parity
     */
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;           /* data bits: 8 */
    options.c_cflag &= ~CSTOPB;    /* stop bit: 1 */
    options.c_cflag &= ~PARENB;    /* disable output parity generation */
    options.c_iflag &= ~INPCK;     /* disable input parity checking */
    options.c_cflag &= ~CRTSCTS;   /* no hw flow control */

    /*
     *  CR & LN
     */
    options.c_iflag &= ~IGNCR;
    options.c_iflag &= ~ICRNL;
    options.c_iflag &= ~INLCR;

    /*
     *  RAW mode
     */
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);  /*Input*/
    options.c_oflag &= ~OPOST;   /*Output*/

    status = tcsetattr(fd_tty, TCSANOW, &options);

    if (status != 0) {
        fprintf(stderr, "'%s': %d, %s\n",
            tty_dev_name, errno, strerror(errno));
        fprintf(stderr, "tcsetattr error\n");
        exit(EXIT_FAILURE);
    }
}

static void tty_cmd(char* cmd)
{
    char cmd_buf[128];
    //char cmd_resp[128];
    char cmd_resp_end[] = { 0x0D, 0x0A, 0x0D, 0x0A, 0x00 };
    char tmp;
    int cmd_resp_index = 0;

    if (fd_tty > -1) {
        int cmd_resp_buf_len = sizeof(cmd_tof_resp);

        snprintf(cmd_buf, sizeof(cmd_buf), "%s\n", cmd);
        printf("CMD:\n%s\n", cmd_buf);
        write(fd_tty, cmd_buf, strlen(cmd_buf));

        /*
         *  First receive the cmd echo
         */
        while (read(fd_tty, &tmp, 1)) {
            if (tmp == cmd_buf[0]) {
                cmd_tof_resp[0] = tmp;
                read(fd_tty, &cmd_tof_resp[1], strlen(cmd_buf) - 1);
                break;
            }
            else {
                /*
                 * other characters from device
                 * ignore it
                 */
            }
        }

        memset(cmd_tof_resp, 0x0, cmd_resp_buf_len);

        while (read(fd_tty, &tmp, 1)) {
            cmd_tof_resp[cmd_resp_index++] = tmp;
            if (cmd_resp_index == cmd_resp_buf_len) {
                memset(cmd_tof_resp, 0x0, cmd_resp_buf_len);
                cmd_resp_index = 0;
            }

            if (strstr(cmd_tof_resp, cmd_resp_end) != NULL) {
                printf("RESP: %s", cmd_tof_resp);
                break;
            }
        }

    }
    else {
        fprintf(stderr, "fd does not exist\n");
        exit(EXIT_FAILURE);
    }
}

static int tty_read_app_result(void)
{
    char app_result[128];
    char app_result_end[] = { 0x0D, 0x0A, 0x00 };
    char tmp;
    int index = 0, result = -1;

    if (fd_tty > -1) {
        int app_result_buf_len = sizeof(app_result);
        memset(app_result, 0x0, app_result_buf_len);

        while (read(fd_tty, &tmp, 1)) {
            app_result[index++] = tmp;
            if (index == app_result_buf_len) {
                /*
                 * buffer full, clear it anyway
                 */
                memset(app_result, 0x0, app_result_buf_len);
                index = 0;
            }

            if (strstr(app_result, app_result_end) != NULL) {
                result = atoi(app_result);
                break;
            }
        }

    }
    else {
        fprintf(stderr, "fd does not exist\n");
        exit(EXIT_FAILURE);
    }

    return (result);
}

#else

#endif

// xxhh
/*
char *best_open_device(int BASE_ADDR, int SIZE);
char *first_capture_addr;
*/

int main(int argc, char* argv[])
{

    /*======================================================================================
     Map the kernel memory location starting from 0x20000000 to the capture processing
    ========================================================================================*/
    /*
  first_capture_addr = best_open_device(DDR_BASE_ADDRESS, DDR_MAP_SIZE);
  if (first_capture_addr == 0) printf("The error locate memory\n");
  else printf("Memory mapped at address %x \n", first_capture_addr);
*/
//udp net

#ifdef net_udp
    struct ifreq ifr;
    const char* iface = "eth0";
    unsigned char* mac;

    struct sockaddr_in server;//, client;
    int  sockfd = 0;
    socklen_t addr_len = sizeof(struct sockaddr_in);
    char buffer_net[200];
    printf("To be connect \n");

    ifr.ifr_addr.sa_family = AF_INET;
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    ioctl(sockfd, SIOCGIFHWADDR, &ifr);
    mac = (unsigned char*)ifr.ifr_hwaddr.sa_data;
    // printf("Mac : %.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n" , mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    bzero(&server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(UDP_SERVER_PORT);
    //server.sin_addr.s_addr = inet_addr(argv[1]);
    server.sin_addr.s_addr = htonl(INADDR_ANY);
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket");
        exit(1);
    }

    if (bind(sockfd, (struct sockaddr*)&server, sizeof(struct sockaddr)) < 0) {
        cout << "connect server error";
        exit(1);
    }
#endif        

#ifdef IP_CAMERA
    struct sockaddr_in server;//, client;
    int  sockfd = 0;
    //socklen_t addr_len = sizeof(struct sockaddr_in);

    bzero(&server, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(TCP_SERVER_PORT);
    server.sin_addr.s_addr = inet_addr("127.0.0.1"); //htonl(INADDR_ANY);
    //SOCKET_STREAM:TCP SOCK_DGRAM:UDP
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        cout << "socket error\n";
        exit(1);
    }
    //Connects to the local server
    /*
    if(connect(sockfd ,(struct sockaddr *)&server, sizeof(server)) <0)
    {
         perror("Connect failed. Error");
         return 1;
    }
    */
    int tmp_net = 0;
    int tmp_net_count = 0;
    do {
        tmp_net = connect(sockfd, (struct sockaddr*)&server, sizeof(server));
        if (tmp_net < 0) {
            sleep(10);
            tmp_net_count = tmp_net_count + 1;
        }
    } while ((tmp_net_count < 100) & (tmp_net < 0));
    if (tmp_net_count > 100) {
        cout << "The error connect\n";
        exit(1);
    }
#endif

    //
 //   errno_t err;
    char gis_filename[60] = "GIS ";
    char ir_filename[80];
    char date_name[8];
    char time_name[6];
#ifdef LOG_FILE
    char filename_log[80];
#endif
#ifdef SAVE_JPG
    char image_name[80];
    char image_name_init[80];
#endif
#ifdef BED_LINE_DEBUG_FILE

    char filename_bed_line_log[80];
#endif
#ifdef BED_COL_DEBUG_FILE
    char filename_bed_col_log[80];
#endif
#ifdef VIDEO_RECORD
#ifdef RESULT_VIDEO
    char avi_filename[80];
#endif
    char avi_source[80], avi_source_ir[80];
#endif


    struct tm* t;
    time_t now;// = time(0);
    int year_int;
    int month_int;
    int day_int;
    int hour_int;
    int min_int;
    int sec_int;
    //  int sec_int_pre = 0;
    std::string year_s;
    std::string month_s;
    std::string day_s;
    std::string hour_s;
    std::string min_s;
    std::string sec_s;

    std::string tmp_string;
#ifdef HTTP_ENABLE
    int send_touchlife_no_pre = 4;
#endif
#ifdef CAMERA_MODE
#ifdef SAVE_DATA
    FILE* fw = NULL;
    FILE* fw_ir = NULL;
#endif    
#ifdef LOG_FILE
    FILE* fw_log;
#endif

    char file_name[80], file_name_init[80];

    bool wotking_camera = 1;
#ifdef DEBUG_PRINT
    int repeat_no = 0;
#endif      
#ifdef net_udp
    struct timeval tv;
    tv.tv_sec = 1;  //  �`�N��core
    tv.tv_usec = 0; // 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    uint64_t receive_buf_size = 1 * 1024 * 1024;  //1 MB
    uint64_t send_buf_size = 100 * 1024 * 1024;  //100 MB
    setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &receive_buf_size, sizeof(receive_buf_size));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &send_buf_size, sizeof(send_buf_size));
    ///////////////////////////////////////////////////////////////////
#endif

    do {
        //  cout << "++++++++++++++++++++++++++++++++++++++++++\n";
        //  cout << "wait the client\n";      
        char msgX[200] = { 0 };
        int cnt1 = 0;
        int iResult = 0;
        int number_err = 0;
        bool receive_head = 0;
#ifdef net_udp      
        sprintf(buffer_net, "Hugo_FallXINIT");
        do {
            cnt1 = recvfrom(sockfd, msgX, 100, 0, (struct sockaddr*)&server, &addr_len);
            if ((cnt1 > 0) & (strcmp(msgX, "Hugo_Fall_INIT") == 0)) {
                receive_head = 1;
                iResult = sendto(sockfd, buffer_net, strlen(buffer_net) + 1, 0, (struct sockaddr*)&server, addr_len);
            }
            number_err++;
            if ((number_err % 10) == 0)cout << "recv error " << number_err << endl;
            sleep(1);

            //  sleep(1);          		
        } while (!receive_head);
        cout << receive_head << " and " << msgX << endl;
#endif      
#ifdef DEBUG_PRINT
        printf("repeat no : %d\n", repeat_no);
        repeat_no++;
#endif
        time(&now);
        t = localtime(&now);
        //char* year, * date, time*;
        year_int = (t->tm_year + 1900);
        month_int = (t->tm_mon + 1);
        day_int = (t->tm_mday);
        hour_int = t->tm_hour;
        min_int = t->tm_min;
        sec_int = t->tm_sec;
        year_s = to_string(year_int);
        month_s = to_string(month_int);
        if (month_int < 10)  month_s = "0" + month_s;
        day_s = to_string(day_int);
        if (day_int < 10) day_s = "0" + day_s;
        hour_s = to_string(hour_int);
        if (hour_int < 10) hour_s = "0" + hour_s;
        min_s = to_string(min_int);
        if (min_int < 10) sec_s = "0" + min_s;
        sec_s = to_string(sec_int);
        if (sec_int < 10) min_s = "0" + sec_s;

        //open date file
        strcpy(file_name, "/home/hugoliu/work/fall_bed/data/");
        strcat(file_name, &year_s[0]);
        strcat(file_name, &month_s[0]);
        strcat(file_name, &day_s[0]);
        strcpy(file_name_init, file_name);
        strcat(file_name, &hour_s[0]);
        strcat(file_name, &min_s[0]);
        strcat(file_name, &sec_s[0]);
#ifdef LOG_FILE
        strcpy(filename_log, file_name);
        strcat(filename_log, ".log");
#endif
        strcpy(gis_filename, file_name);
        /*
        if ((_access("data", 0)) == -1) {
            system("mkdir data");
        }
        */
#ifdef LOG_FILE
        if ((err = fopen_s(&fw_log, filename_log, "w")) != 0) {
            printf("Unable to write log file \"%s\"\n", filename_log);
        }
#endif

#else
    bool IS_IR = 1;
    bool IS_NEW = 1;

    FILE* fr;
    FILE* fr_ir;
#ifdef LOG_FILE
    FILE* fw_log;
#endif
#ifdef BED_LINE_DEBUG_FILE
    FILE* fw_row_log;
#endif
#ifdef BED_COL_DEBUG_FILE
    FILE* fw_col_log;
#endif
    if (argc < 3) {
        printf("No input Binary File or timedlay\n");
        exit(-1);
    }
    fr = fopen(argv[1], "rb");
    if (fr == NULL) {
        printf("Unable to open tof fileX \"%s\"\n", argv[1]);
        exit(-1);
    }
    /*
        if ((err = fopen(&fr, argv[1], "rb")) != 0) {
            printf("Unable to open tof file \"%s\"\n", argv[1]);
            exit(-1);
        }
    */
    strcpy(ir_filename, argv[1]);

    strcpy(gis_filename, ir_filename);


#ifdef LOG_FILE
    strcpy(filename_log, ir_filename);
#endif
#ifdef Compare_LOG
    char comapre_log[80];
    FILE* ComPfw_log;
    strcpy(comapre_log, gis_filename);
    size_t log_str_len = strlen(comapre_log);
    comapre_log[log_str_len - 3] = '\0';
    strcat(comapre_log, "_comp.log");
    if ((err = fopen_s(&ComPfw_log, comapre_log, "w")) != 0) {
        printf("Unable to write log file \"%s\"\n", comapre_log);
    }
#endif
#ifdef BED_LINE_DEBUG_FILE
    strcpy(filename_bed_line_log, gis_filename);
    strcat(filename_bed_line_log, "_bed_line.csv");
#endif

#ifdef SAVE_JPG
    strcpy(image_name, gis_filename);
    strcat(image_name, ".jpg");
    strcpy(image_name_init, gis_filename);
    strcat(image_name_init, "_init.jpg");
#endif
#ifdef BED_COL_DEBUG_FILE

    strcpy(image_name, gis_filename);
    strcat(image_name, ".jpg");
    strcpy(filename_bed_col_log, gis_filename);
    strcat(filename_bed_col_log, "_bed_col.csv");
#endif
    strcat(ir_filename, "_ir");
#ifdef LOG_FILE
    strcat(filename_log, ".log");
#endif


    printf("The ir file name is %s\n", ir_filename);
    fr_ir = fopen(ir_filename, "rb");
    if (fr_ir == NULL) {
        printf("Unable to open IR file \"%s\"\n", ir_filename);
        IS_IR = 0;
        //exit(-1);
    }

#ifdef LOG_FILE
    if ((err = fopen_s(&fw_log, filename_log, "w")) != 0) {
        printf("Unable to write log file \"%s\"\n", filename_log);

    }
#endif
#ifdef BED_LINE_DEBUG_FILE
    if ((err = fopen_s(&fw_row_log, filename_bed_line_log, "w")) != 0) {
        printf("Unable to write row log file \"%s\"\n", filename_bed_line_log);
    }
#endif
#ifdef BED_COL_DEBUG_FILE
    if ((err = fopen_s(&fw_col_log, filename_bed_col_log, "w")) != 0) {
        printf("Unable to write col log file \"%s\"\n", filename_bed_col_log);
    }
#endif

#endif
#ifdef VIDEO_RECORD
    //  char avi_filename[80], avi_source[80], avi_source_ir[80];

#ifdef CAMERA_MODE

    strcpy(avi_source, file_name);
    strcpy(avi_source_ir, file_name);
#else
    strcpy(avi_source, argv[3]);
    strcpy(avi_source_ir, argv[3]);
    size_t avi_str_len = strlen(avi_source);

    avi_source[avi_str_len - 3] = '\0';
    avi_source_ir[avi_str_len - 3] = '\0';
#endif
    strcat(avi_source, "_ti.avi");
    strcat(avi_source_ir, "_ir.avi");

#ifdef RESULT_VIDEO
    strcpy(avi_filename, argv[3]);
    avi_filename[avi_str_len - 3] = '\0';
    strcat(avi_filename, ".mp4");
    VideoWriter writer(avi_filename, CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(640, 480));
#endif
    VideoWriter writer_source;
    int ex = CV_FOURCC('M', 'J', 'L', 'S');
    //    int ex = CV_FOURCC('D', 'X', '5', '0');
        //int ex = CV_FOURCC('M', 'J', 'P', 'G');
        // Transform from int to char via Bitwise operators
        //char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };
    writer_source.open(avi_source, ex, 25.0, Size(80, 60), 0);
    VideoWriter writer_ir(avi_source_ir, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(80, 60), 0);
    writer_source.set(VIDEOWRITER_PROP_QUALITY, 100.0);
    writer_ir.set(VIDEOWRITER_PROP_QUALITY, 100.0);
    //    VideoWriter writer(avi_filename, CV_FOURCC('M', 'P', '4', '2'), 25.0, Size(640, 480));
//    VideoWriter writer(avi_filename, CV_FOURCC('M', 'J', 'P', 'G'), 25.0, Size(640, 480));
#endif

    int fall_th;
    cv::String String_tmp;
    int edge_level = 5;
#ifndef CAMERA_MODE
    char head_title[9];

    size_t result, result_ir;

    //    unsigned char* total_len;
    unsigned char* bufferX;
    unsigned char* buffer_head;
    unsigned char* buffer_irX;
    unsigned char* buffer_head_ir;
    //    buffer_head = (unsigned char*)malloc(sizeof(unsigned char) * 8);
    bufferX = (unsigned char*)malloc(sizeof(unsigned char) * TOF_DEPTH_PIXELS);
    buffer_head = (unsigned char*)malloc(sizeof(unsigned char) * 8);
    buffer_irX = (unsigned char*)malloc(sizeof(unsigned char) * TOF_DEPTH_PIXELS);
    buffer_head_ir = (unsigned char*)malloc(sizeof(unsigned char) * 8);
#endif
#ifndef net_udp
#ifndef IP_CAMERA
    namedWindow(gis_filename, WINDOW_AUTOSIZE); //WINDOW_OPENGL); // CV_WINDOW_AUTOSIZE);
#endif    
#endif    

//    namedWindow("controller",  WINDOW_OPENGL);
    //resizeWindow(gis_filename, 640, 480);
    //    resizeWindow("controller", 400, 100);
    //    resizeWindow("controller", 400, 120);
#ifdef net_udp 
    Mat show_image = Mat(240, 640, CV_8UC3, Scalar(0));
    Mat show_image_original = Mat(60, 160, CV_8UC3, Scalar(0));
    Mat image_send_mono = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat image_send = Mat(60, 80, CV_8UC3, Scalar(0));
    Mat image_ir_mono = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat image_ir_send = Mat(60, 80, CV_8UC3, Scalar(0));
#else
    Mat show_image = Mat(480, 640, CV_8UC3, Scalar(0));
    Mat show_image_nono = Mat(480, 640, CV_8UC1, Scalar(0));
#endif    
    Mat Image_mask = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat image_last = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat depth8U = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat depth8U_ir = Mat(60, 80, CV_8UC1, Scalar(0));

    //  Mat grayImage_XX_rise = Mat(240, 320, CV_8UC1, Scalar(0));
    Mat C_grayImage_XX_rise = Mat(240, 320, CV_8UC3, Scalar(0));
    Mat grayImage_GIS = Mat(240, 320, CV_8UC3, Scalar(0));
    Mat C_grayImage_4X_rise = Mat(240, 320, CV_8UC3, Scalar(0));
    Mat image_pre_fall = Mat(60, 80, CV_8UC1, Scalar(0));


    Mat grayImage_now, grayImage_now_th;
    Mat grayImage_pre, grayImage_pre_pre;
    Mat grayImage_init = Mat(60, 80, CV_8UC1, Scalar(0));
    Mat grayImage_initX = Mat(60, 80, CV_8UC1, Scalar(0));

    //  Mat resize_img;
#ifndef net_udp
    Mat resize_img_ir;
    Mat C_resize_img_ir, C_image_last_rise;
#endif
    int tmpMin = 0;

    bool initial_begin = 1;
    int now_x = 0, now_y = 0, pre_x = 0, pre_y = 0;
    int keep_x = 0, keep_y = 0;
    int keep_old_x = 0, keep_old_y = 0;
    int keep_old_value = 0;
    int keep_first_stage_x = 0, keep_first_stage_y = 0;
    int keep_first_stage_end_x = 0, keep_first_stage_end_y = 0;
    //other people
    int other_people_x = 0;
    int other_people_y = 0;
    int other_people_value = 0;
    int other_people_now = 0;
    //right and left
    bool first_change = 1;
    bool turn_up = 0;
    bool turn_up_pre = 0;
    bool trun_lie_down = 0; //躺下
    bool lie_down_mode = 0;
    int bed_first_line = 0;
    //   int bed_half_line = 0;
    int bed_end_line = 0;
    int bed_width = 0;
    int keep_equal_same_up = 0;
    bool bed_edge = 0;
    int bed_high = 0;
    int no_bed = 1;
    bool fall_hold = 0;
    bool fall_true = 0;
    bool real_bed_mode = 0;

    bool other_people_inside = 0;
    bool other_people_inside_pre = 0;
#ifndef net_udp 
    bool show_people_high = 0;
#endif
    bool no_people = 0;
    bool no_people_pre = 0;
    int people_frame = 120;
    int people_here_count = 0;
    bool have_people = 0;
    bool have_people1 = 0;
    bool inside_bed = 0;
    bool inside_bed_pre = 0;
    bool is_inside_bed = 0;
    bool inside_bed_tmp = 0;
    int col_total[80];
    int line_total[60];
    int line_half[60];
    int line_total_init[60];
    int line_half_init[60];
    int col_total_init[80];
    int bed_count_number = 0;
    int first_line = 0;
    int first_value = 0;
    int bed_count = 0;
    int row_high = 0;
    int col_high = 0;
    int col_width = 0;
    int col_end_line = 0;
    int col_first_line = 0;
    Bed bed_area{}, bed_area_pre{}, bed_area_init{};
    bool leave_bed = 0;
    bool leave_bed_real = 0;
    char tmp_status[9];
    bool edge_lie_down = false;
    bool up_lie_down = false;
    bool is_lie_down = 0;

    bool lie_down_ready = 0;
    //bool bed_second_method = 0;
    //20200802
    int edge_up_limit = 5;
    int edge_down_limit = 6;
    //int lie_down_high = 13;
    //20200804
    int people_height = 0, people_height_pre = 0, people_height_pre2 = 0;
    //20200805
  //  int up_center_x = 0;
  //  int up_center_y = 0;
    int bed_height = 0;
    //20200806 解決固定高度
    int box_area = 0;
    int box_area_now = 0;
    int box_area_pre = 0;
    int box_area_now_pre = 0;

    //20200811
    //int up_diff_th = 10;
    int up_high_level = 28;
    int down_level_th = 15;
    //to correct people 20200826
    bool people_counnt_enable = 1;
    //20200827 To fix have_people or no_people
    //bool no_people_act = 0;
    bool have_people_act = 0;
    //20200828
    bool sick_house = 0;
    //20200903
    int max_hight_init = 0;
    //20200923 for pre
    bool other_people2 = 0;
    bool singal_person = 0;
    bool max_high_less_240 = 0;
    //20220120
    bool EdgeStatus = false; //2022 "To distinct the behavior of "UP" and "Edge"
    int HumanOnEdgeLineCount = 0; //2022 To Count How Many Times Human On Edge
    int GroundAvgUpPixel = 0; //2022 
    int GroundAvgDownPixel = 0; //2022
    //int Pre_AvgPixel_Inbed = 0;//2022 To improve the object leaving in bed by human
    bool bed_obj_status = false; //20220126
    int bed_status = 0; //20220126

#ifdef    SHOW_MULTI_PEOPLE
    int people_number = 0;
#endif
    bool now_in_bed = 0;

    //
    int draw_frame = 0;
    int contour_area_now = 0;
    char text_show[30];
#ifdef NO_FRAME_NO
    int frame_no_is = 0;
#else
    int frame_no_is = 1;
#endif
    Rect bounding_rect_now;
#ifdef ENGINEER
    bool is_engineer = 1;
#else
    bool is_engineer = 0;
#endif
    int display_bed_detect = 0;
#ifdef display_message
    display_bed_detect = 1;
#endif
    int show_fall_message = 0;
    int show_fall_message_pre = 0;
#ifdef show_message
    show_fall_message = 1;
#endif
#ifdef show_message_pre
    show_fall_message_pre = 1;
#endif
    bool game_bed_fall_over = 0;
    int frame_no = 0;
#ifdef CAMERA_MODE
    //  char* get_frame_cmd;
   //   get_frame_cmd = (char*)CMD_CAMERA_GET_H_DEPTH_IR_FRAME;

    bool header_begin = false;
    bool go_active = false;
    bool go_active_pre = false;
    bool go_init = false;
    bool new_frame = false;
    Mat image_0, image_1, image_2;
    Mat image_ir_0, image_ir_1, image_ir_2;
#ifndef net_udp    
#ifndef IP_CAMERA
    cout << "\n" << file_name_init << "\n";
    printf(" press g to go \"q\" or Esc to leave\n");
#endif    
#endif    

    static int iWaitKey = 0;
    ///////////////////////////////////////////////////////////////////
    // initial tof camera    modify by hugo liu 20220124
    uvc_dev_name = (char*)"/dev/video0";
    tty_dev_name = (char*)"/dev/ttyUSB0";
    //char cmd_tof[128];
    open_tty_device();
    /*
    config device
    */
    //sprintf(cmd_tof, ); 
    tty_cmd("get product_sn");
    cout << "Tod serial No: " << cmd_tof_resp << " over\n";
    close_tty_device();
    char tof_serial[10];
    for (int i = 15; i < 27; i++)
        tof_serial[i - 15] = cmd_tof_resp[i];

    //    cout << "The V4L Camera to setup\n";
    open_uvc_device();
    init_device();
    start_capturing();
    cout << "The V4L Camera setup ok" << uvc_dev_name << "\n";
    /////////////////////////////////////////////////////////
    char frame_title_end[8];
    const char frame_head[8] = "%s%2d";
    cout << "the enter depth \n";
    FILE* fw_frame0 = NULL;
    FILE* fw_frame1 = NULL;
    FILE* fw_frame2 = NULL;
    unsigned char* bufferX;
    bufferX = (unsigned char*)malloc(sizeof(unsigned char) * TOF_DEPTH_PIXELS);;
    FILE* fr_frame0 = NULL;
    FILE* fr_frame1 = NULL;
    FILE* fr_frame2 = NULL;
    bool init_data = 0;
    int result_init = 0;
    //  go_active = 1;
    ////////////////////////////////////////////////////////////////////////////////////////    
    while ((iWaitKey != 27) && (iWaitKey != 'q') && (game_bed_fall_over == 0))
    {
        new_frame = read_frame();
        if (new_frame) {
#ifdef DEBUG_PRINT            
            printf("get new drame\n");
#endif            
            new_frame = false;
            /*
            memcpy(depth.ptr<unsigned short>(0), depth_ir_fb, DEPTH_FRAME_SIZE);
            memcpy(ir.ptr<unsigned short>(0), &depth_ir_fb[DEPTH_FRAME_SIZE], DEPTH_FRAME_SIZE);
            depth.convertTo(image, CV_8UC1, 255.0 / 4096);
            ir.convertTo(image_ir, CV_8UC1, 255.0 / 1024);
            */
            tof_depth.convertTo(image, CV_8UC1, 255.0 / 4096);
            tof_ir.convertTo(image_ir, CV_8UC1, 255.0 / 1024);
            /////////////////////////////////////////

            if (!go_active & !go_init) {
#ifndef net_udp            
                resize(image, resize_img, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
                resize(image_ir, resize_img_ir, Size(image_ir.cols * 4, image_ir.rows * 4), 0, 0, INTER_LINEAR);
#endif                   
                //  printf("The show frame no %d\n", frame_no);
                frame_no = frame_no + 1;
#ifdef net_udp         
                cvtColor(image, image_send, CV_GRAY2RGB);
                cvtColor(image_ir, image_ir_send, CV_GRAY2RGB);
                image_send.copyTo(show_image_original(Rect(0, 0, 80, 60)));
                image_ir_send.copyTo(show_image_original(Rect(80, 0, 80, 60)));
                //sprintf(text_show, " ");
#else                
                if (init_data == 1) sprintf(text_show, "init data ");
                else sprintf(text_show, "          ");
                cvtColor(resize_img, C_grayImage_4X_rise, CV_GRAY2RGB);
                cvtColor(resize_img_ir, C_resize_img_ir, CV_GRAY2RGB);
                String_tmp = IMAGE_PATH;
                String_tmp = String_tmp + "gis_320x240.jpg";
                C_grayImage_XX_rise = imread(String_tmp);
                if (!C_grayImage_XX_rise.data) {
                    C_grayImage_XX_rise = Mat(240, 320, CV_8UC3, Scalar(0));
                }
                C_grayImage_XX_rise.copyTo(show_image(Rect(0, 240, 320, 240)));
                C_grayImage_XX_rise.copyTo(show_image(Rect(320, 240, 320, 240)));
                C_grayImage_4X_rise.copyTo(show_image(Rect(0, 0, 320, 240)));
                C_resize_img_ir.copyTo(show_image(Rect(320, 0, 320, 240)));
#ifndef IP_CAMERA                
                imshow(gis_filename, show_image);
#endif                
#endif                
            } //(!go_active)
            else if (go_init) {
                if (frame_no < 3) {
                    time(&now);
                    t = localtime(&now);
                    //char* year, * date, time*;
                    year_int = (t->tm_year + 1900);
                    month_int = (t->tm_mon + 1);
                    day_int = (t->tm_mday);
                    hour_int = t->tm_hour;
                    min_int = t->tm_min;
                    year_s = to_string(year_int);
                    month_s = to_string(month_int);
                    if (month_int < 10)  month_s = "0" + month_s;
                    day_s = to_string(day_int);
                    if (day_int < 10) day_s = "0" + day_s;

                    hour_int = t->tm_hour;
                    min_int = t->tm_min;
                    sec_int = t->tm_sec;
                    hour_s = to_string(hour_int);
                    if (hour_int < 10) hour_s = "0" + hour_s;
                    min_s = to_string(min_int);
                    if (min_int < 10) min_s = "0" + min_s;
                    sec_s = to_string(sec_int);
                    if (sec_int < 10) sec_s = "0" + sec_s;

                    strcpy(date_name, &year_s[0]);
                    strcat(date_name, &month_s[0]);
                    strcat(date_name, &day_s[0]);

                    strcpy(time_name, &hour_s[0]);
                    strcat(time_name, &min_s[0]);
                    strcat(time_name, &sec_s[0]);
                }
                if (frame_no == 0) {
                    //fw = fopen(file_name, "wb");
                    //String_tmp = WORK_PATH;
                    //String_tmp = String_tmp + "init/frame0.ti";
                    //char str0[50] = WORK_PATH;                    
                    //strcat(str0, "init/frame0.ti");
                    fw_frame0 = fopen("./init/frame0.ti", "wb");
                    if (fw_frame0 == NULL) {
                        printf("Unable to write TOF file \"%s\"\n", "frame0.ti");
                        exit(1);
                    }
                    fprintf(fw_frame0, "HugoNano");
                    fprintf(fw_frame0, "%s", date_name);
                    fwrite(image.data, sizeof(unsigned char), 80 * 60, fw_frame0);
                    fclose(fw_frame0);
                }
                if (frame_no == 1) {
                    //char str1[50] = WORK_PATH;                    
                    //strcat(str1, "init/frame1.ti");
                    fw_frame1 = fopen("./init/frame1.ti", "wb");
                    if (fw_frame1 == NULL) {
                        printf("Unable to write TOF file \"%s\"\n", "frame1.ti");
                        exit(1);
                    }
                    fprintf(fw_frame1, "HugoNano");
                    fprintf(fw_frame1, "%s", date_name);
                    fwrite(image.data, sizeof(unsigned char), 80 * 60, fw_frame1);
                    fclose(fw_frame1);
                }
                if (frame_no == 2) {
                    //char str2[50] = WORK_PATH;                    
                    //strcat(str2, "init/frame2.ti");
                    fw_frame2 = fopen("./init/frame2.ti", "wb");
                    if (fw_frame2 == NULL) {
                        printf("Unable to write TOF file \"%s\"\n", "frame2.ti");
                        exit(1);
                    }
                    fprintf(fw_frame2, "HugoNano");
                    fprintf(fw_frame2, "%s", date_name);
                    fwrite(image.data, sizeof(unsigned char), 80 * 60, fw_frame2);
                    fclose(fw_frame2);
                    //go_init = 0;
                }
                // sprintf(text_show, date_name);
                resize(image, show_image_nono, Size(image.cols * 8, image.rows * 8), 0, 0, INTER_LINEAR);
                cvtColor(show_image_nono, show_image, CV_GRAY2RGB);
                // putText(show_image, text_show, Point(10, 20), FONT_HERSHEY_TRIPLEX, 1.0, Scalar(18, 12, 255), 1, 8);
                 //C_grayImage_4X_rise.copyTo(show_image(Rect(0, 0, 320, 240)));
                frame_no = frame_no + 1;
            }
            else
            {
                if (frame_no == 0) {
                    /*
                      char str0[50] = WORK_PATH;
                      strcat(str0, "init/frame0.ti");
                      */
                    fr_frame0 = fopen("./init/frame0.ti", "rb");
                    if (fr_frame0 == NULL) {
                        printf("Unable to open tof file frame0.ti\"\n");
                        init_data = 0;
                    }
                    else {
                        init_data = 1;
                    }
                    /*
                    char str1[50] = WORK_PATH;
                    strcat(str1, "init/frame1.ti");
                    */
                    fr_frame1 = fopen("./init/frame1.ti", "rb");
                    if (fr_frame1 == NULL) {
                        printf("Unable to open tof file frame1.ti\"\n");
                        init_data = 0;
                    }
                    else {
                        init_data = 1;
                    }
                    /*
                    char str2[50] = WORK_PATH;
                    strcat(str2, "init/frame2.ti");
                    */
                    fr_frame2 = fopen("./init/frame2.ti", "rb");
                    if (fr_frame2 == NULL) {
                        printf("Unable to open tof file frame2.ti\"\n");
                        init_data = 0;
                    }
                    else {
                        init_data = 1;
                    }
                    //  printf("=========Go read file==========%d \n", init_data);
                }
                if ((init_data == 1) & (frame_no < 3)) {
                    if (frame_no == 0) {
                        fseek(fr_frame0, 16, SEEK_SET);
                        result_init = fread(bufferX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr_frame0);
                        fclose(fr_frame0);
                        //cout << "frame no 0 " << result_init << "\n";
                    }
                    else if (frame_no == 1) {
                        fseek(fr_frame1, 16, SEEK_SET);
                        result_init = fread(bufferX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr_frame1);
                        fclose(fr_frame1);
                        //cout << "frame no 1\n";
                    }
                    else if (frame_no == 2) {
                        fseek(fr_frame2, 16, SEEK_SET);
                        result_init = fread(bufferX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr_frame2);
                        fclose(fr_frame2);
                        //cout << "frame no 2\n";
                    }
                    /*
                    else
                          free(bufferX);
                     if (frame_no <3)  {
                         */
                    memcpy(image.ptr<unsigned short>(0), bufferX, TOF_DEPTH_PIXELS);
                    //   resize(image, resize_img, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
//                     }

                }
                //#ifndef net_udp            
                //            resize(image, resize_img, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
                           // resize(image_ir, resize_img_ir, Size(image_ir.cols * 4, image_ir.rows * 4), 0, 0, INTER_LINEAR);
                //#endif  
                                //cout << "debug0\n";

#ifdef SAVE_DATA                  
              //  cout << "debug1 \n"
                time(&now);
                t = localtime(&now);
                //char* year, * date, time*;
                year_int = (t->tm_year + 1900);
                month_int = (t->tm_mon + 1);
                day_int = (t->tm_mday);
                hour_int = t->tm_hour;
                min_int = t->tm_min;
                year_s = to_string(year_int);
                month_s = to_string(month_int);
                if (month_int < 10)  month_s = "0" + month_s;
                day_s = to_string(day_int);
                if (day_int < 10) day_s = "0" + day_s;

                hour_int = t->tm_hour;
                min_int = t->tm_min;
                sec_int = t->tm_sec;
                hour_s = to_string(hour_int);
                if (hour_int < 10) hour_s = "0" + hour_s;
                min_s = to_string(min_int);
                if (min_int < 10) min_s = "0" + min_s;
                sec_s = to_string(sec_int);
                if (sec_int < 10) sec_s = "0" + sec_s;

                strcpy(date_name, &year_s[0]);
                strcat(date_name, &month_s[0]);
                strcat(date_name, &day_s[0]);

                strcpy(time_name, &hour_s[0]);
                strcat(time_name, &min_s[0]);
                strcat(time_name, &sec_s[0]);
                /***************************************************************************/
                //fall detect
                /***************************************************************************/
                if ((frame_no % 600000) == 0) {
                    if (frame_no != 0) {
                        fseek(fw, 8, SEEK_SET);
                        fprintf(fw, "%8d", frame_no);
                        fclose(fw);
                        fseek(fw_ir, 8, SEEK_SET);
                        fprintf(fw_ir, "%8d", frame_no);
                        fclose(fw_ir);
                        //          cout << "The new file " << frame_no << "\n";
                    }

                    //open date file
                    //file_name[0] = '\0';
                    strcpy(file_name, file_name_init);
                    strcat(file_name, &hour_s[0]);
                    strcat(file_name, &min_s[0]);
                    strcat(file_name, &sec_s[0]);
                    strcat(file_name, ".ti");
                    fw = fopen(file_name, "wb");
                    if (fw == NULL) {
                        printf("Unable to write TOF file \"%s\"\n", file_name);
                        exit(1);
                    }
                    //     cout << file_name << "\n";
                    strcpy(ir_filename, file_name);
                    strcat(ir_filename, "_ir");
                    fw_ir = fopen(ir_filename, "wb");
                    if (fw_ir == NULL) {
                        printf("Unable to write IR file \"%s\"\n", ir_filename);
                        exit(-1);
                    }
                    header_begin = false;
                } //frame_no % 600000
                // add the header file
                if (!header_begin) {
                    fprintf(fw, "HugoNano");
                    //fprintf(fw, "%8d", frame_no);

                   // sprintf(frame_title_end, frame_head, date_name, frame_no);
                    fprintf(fw, "%s", date_name);

                    fprintf(fw_ir, "IR  Nano");
                    fprintf(fw_ir, "%s", date_name);
                    //fprintf(fw_ir, "%8c", frame_title_end);
                    sprintf(frame_title_end, frame_head, time_name, frame_no);
                    if (frame_no > 0) {
                        fprintf(fw, "%s", frame_title_end);
                        fwrite(image_0.data, sizeof(unsigned char), 80 * 60, fw);
                        //fprintf(fw, "       1");
                        fprintf(fw, "%s", frame_title_end);
                        fwrite(image_1.data, sizeof(unsigned char), 80 * 60, fw);
                        fprintf(fw, "%s", frame_title_end);
                        fwrite(image_2.data, sizeof(unsigned char), 80 * 60, fw);

                        fprintf(fw_ir, "       0");
                        fwrite(image_ir_0.data, sizeof(unsigned char), 80 * 60, fw_ir);
                        fprintf(fw_ir, "       1");
                        fwrite(image_ir_1.data, sizeof(unsigned char), 80 * 60, fw_ir);
                        fprintf(fw_ir, "       2");
                        fwrite(image_ir_2.data, sizeof(unsigned char), 80 * 60, fw_ir);
                        //grayImage_now
                    }
                } //!head
                header_begin = true;
                int frame_no_rmp = frame_no % 100;
                string frame_s = to_string(frame_no_rmp);
                if (frame_no_rmp < 10) frame_s = "0" + frame_s;
                char time_name_tmp[8];
                strcpy(time_name_tmp, time_name);
                strcat(time_name_tmp, &frame_s[0]);
                //sprintf(frame_title_end, frame_head, time_name, frame_no);
                fprintf(fw, "%s", time_name_tmp);
                //cout << frame_title_end << "\n";
                fwrite(image.data, sizeof(unsigned char), 80 * 60, fw);
                //////////////
                fprintf(fw_ir, "%8c", frame_title_end);
                // cout << frame_title_end << "\n";
                fwrite(image_ir.data, sizeof(unsigned char), 80 * 60, fw_ir);
#endif                
                //  cout << "frame_no: "  << frame_no << "\n";
#else //no Camera
    bool go_active = true;
    uchar iKey;
    result = fread(&head_title, sizeof(unsigned char), 8, fr);

    if (strcmp(head_title, "Hugo8060") == 0) {
        result = fread(buffer_head, sizeof(unsigned char), 8, fr);
        buffer_head[8] = '\0';
        IS_NEW = 1;
        printf("The total frame is %s\n", buffer_head);
    } /*
    else if (strcmp(head_title, "HugoNano") == 0) {
        result = fread(buffer_head, sizeof(unsigned char), 8, fr);
        buffer_head[8] = '\0';
        printf("The total frame is %s\n", buffer_head);
        IS_NEW = 0;
    } */
    else IS_NEW = 0;
    printf("The file is new? %d and head %s\n", IS_NEW, buffer_head);

    while (!game_bed_fall_over) {
        frame_no = 0;

        fseek(fr, 16, SEEK_SET);
        result = fread(buffer_head, sizeof(unsigned char), 8, fr);
        result = fread(bufferX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr);
        if (IS_IR) {
            fseek(fr_ir, 16, SEEK_SET);
            fread(buffer_head_ir, sizeof(unsigned char), 8, fr_ir);
            result_ir = fread(buffer_irX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr_ir);
        }
        do {  //result
            memcpy(image.ptr<unsigned short>(0), bufferX, TOF_DEPTH_PIXELS);
            memcpy(image_ir.ptr<unsigned short>(0), buffer_irX, TOF_DEPTH_PIXELS);

#endif
#ifdef VIDEO_RECORD
            Mat cimage, cimage_ir, cimage_X;
            //           Mat cimageX, cimage_irX;
            //           cvtColor(image, cimage, CV_GRAY2RGB);
           //            cvtColor(cimage, cimageX, CV_RGB2YUV);
            //           cvtColor(image_ir, cimage_ir, CV_GRAY2RGB);
           //            cvtColor(cimage_ir, cimage_irX, CV_RGB2YUV);
                       /*
                       Laplacian(image, cimage_ir,CV_16S);
                       cimage_ir.convertTo(cimage_X, CV_8U);
                       add(image, cimage_X, cimage);
                       */
            writer_source << image; // cimage;
            writer_ir << image_ir;// cimage_ir;
#endif
            //tof

            Mat_<uchar>  now_image_1 = image;
            //ir
            //cout << "debug1\n";
#ifndef net_udp            
            resize(image_ir, resize_img_ir, Size(image_ir.cols * 4, image_ir.rows * 4), 0, 0, INTER_LINEAR);
            int XX_rise_x = 50;
            int XX_rise_y = 50;
#endif            


            /***************************************************************************/
            //fall detect
            /***************************************************************************/

            max_hight_now = 0;
            int max_depth_tmp = 0;
#ifdef MAX_DEPTH_LOC
            int index_high_x = 0;
            int index_high_y = 0;
#endif
            //max_hight_init = 0;
            int max_depth_tmp_th = 180;
            Mat_<uchar>  ImageMask_1 = Image_mask;
            for (int i = 0; i < image.rows; i = i + 2) {
                for (int j = 0; j < image.cols; j = j + 2) {
                    if ((ImageMask_1(i, j) == 0) && (ImageMask_1(i + 1, j) == 0) && (ImageMask_1(i, j + 1) == 0) && (ImageMask_1(i + 1, j + 1) == 0)) {
                        max_depth_tmp = (now_image_1(i, j) + now_image_1(i + 1, j) + now_image_1(i, j + 1) + now_image_1(i + 1, j + 1)) >> 2;
                        if ((max_hight_now < max_depth_tmp) && (max_depth_tmp <= max_depth_tmp_th) && (Image_mask.at<uchar>(i, j) == 0)) {
                            max_hight_now = max_depth_tmp;
#ifdef MAX_DEPTH_LOC
                            index_high_x = j;
                            index_high_y = i;
#endif
                        }
                    }
                }
            }
#ifdef MAX_DEPTH_LOC
            printf("*****$$$$$ The frame_no %d high is (%d,%d)=%d\n", frame_no, index_high_x, index_high_y, max_hight_now);

#endif
            //cout << "Debug2\n";

            uchar  tmp, tmp1;
            //int max_hight_now_tmp = ((max_hight_now - tmpNowMin) << 4) / 10;
            // printf("The frame_no %d max Depth (%d,%d) = %d\n", frame_no, index_high_x, index_high_y, max_hight_now_tmp);
            tmp = max_hight_now - 4; // 10; //平整
            tmp1 = tmp - P_height; // 20;  //processing 80cm 左右
            grayImage_now = Mat::zeros(image.size(), CV_8UC1);


            /* 去黑 */
            //if (frame_no > 2) {
            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    //                    if (Image_mask.at<uchar>(i, j) == 0) {
                    if (now_image_1(i, j) < 10) {
                        if ((j > 0) && (j < (image.cols - 1)) && (i > 0) && (i < (image.rows - 1))) {
                            if (now_image_1(i, j + 1) > 10) grayImage_now.at<uchar>(i, j) = now_image_1(i, j + 1);
                            else if (now_image_1(i, j - 1) > 10) grayImage_now.at<uchar>(i, j) = now_image_1(i, j - 1);
                            else if (now_image_1(i - 1, j) > 10) grayImage_now.at<uchar>(i, j) = now_image_1(i - 1, j);
                            else if (now_image_1(i + 1, j) > 10) grayImage_now.at<uchar>(i, j) = now_image_1(i + 1, j);
                            else grayImage_now.at<uchar>(i, j) = tmp;
                        }
                        else grayImage_now.at<uchar>(i, j) = tmp;
                    }
                    else if (now_image_1(i, j) > tmp) grayImage_now.at<uchar>(i, j) = tmp;
                    else grayImage_now.at<uchar>(i, j) = now_image_1(i, j);
                    /*
                }
                else grayImage_now.at<uchar>(i, j) = now_image_1(i, j);
                */
                }
            }
            //?????
            //image_send = grayImage_now.clone();
            //image_ir_send = image_ir.clone();
#ifdef net_udp            
            image_send_mono = grayImage_now.clone();
            cvtColor(image_send_mono, image_send, CV_GRAY2RGB);
            image_ir_mono = image_ir.clone();
            cvtColor(image_ir_mono, image_ir_send, CV_GRAY2RGB);
#else            
            resize(grayImage_now, resize_img, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR); //image
            cvtColor(resize_img, C_grayImage_4X_rise, CV_GRAY2RGB);
#endif
            grayImage_now_th = Mat::zeros(grayImage_now.size(), CV_8UC1);
            Mat_<uchar>  now_image = grayImage_now;
            for (int i = 0; i < grayImage_now.rows; i++) {
                for (int j = 0; j < grayImage_now.cols; j++) {
                    if (now_image(i, j) < tmp1)
                        grayImage_now_th.at<uchar>(i, j) = tmp;
                    else grayImage_now_th.at<uchar>(i, j) = now_image(i, j);
                }
            }
            //身高偵測

            if (frame_no < 2) {
                now_x = 0;
                now_y = 0;
            }
            else {
                tmpNowMin = 1000;
                Mat_<uchar>  now_image_2 = grayImage_now;
                int people_high_position_start_x = 24;
                if (no_bed == 1) people_high_position_start_x = 10;
                for (int i = 10; i < (grayImage_now.rows - 10); i++) {
                    for (int j = people_high_position_start_x; j < (grayImage_now.cols - 10); j++) {
                        int people_hi_tmp = ((now_image_2(i, j) + now_image_2(i + 1, j) + now_image_2(i, j + 1) + now_image_2(i + 1, j + 1)) >> 2);
                        if ((now_image_2(i, j) < tmpNowMin) && (abs(people_hi_tmp - now_image_2(i, j)) < 5)
                            && (Image_mask.at<uchar>(i, j) == 0)) {
                            tmpNowMin = grayImage_now.at<uchar>(i, j);
                            now_x = j;
                            now_y = i;
                        }
                    }
                }

            }
#ifndef net_udp             
            int now_4x = (now_x * 4);
            int now_4y = (now_y * 4);
#endif            
            //   Mat grayImage_now_rise;
               // tmpNowMin = tmpxCountMinVal;
                //            int people_height = ((tmpCountMaxVal - grayImage_now.at<uchar>(now_y, now_x)) * 16) / 10;
            int people_height_tmp = ((max_hight_now - tmpNowMin) << 4) / 10;
            people_height = 0;
            if (people_height_tmp > 190) people_height = people_height_pre;
            else people_height = people_height_tmp;

            //  int keep_value = 0;

            int check_point_high = 0;
            if (frame_no > 2) {
                //multi people detect
//multi people detect
#ifdef SHOW_MULTI_PEOPLE
                //grayImage_init  grayImage_now
                Mat gray_people;

                Mat grayImage_init_post, grayImage_now_post;
                threshold(grayImage_init, grayImage_init_post, 90, 255, THRESH_TRUNC); //more high 85
                threshold(grayImage_now, grayImage_now_post, 90, 255, THRESH_TRUNC);
                absdiff(grayImage_init_post, grayImage_now_post, gray_people);

                /*
                Mat grayImage_init_post = grayImage_init | Image_mask;
                Mat grayImage_now_post = grayImage_now | Image_mask;
                absdiff(grayImage_init_post, grayImage_now_post, gray_people);
                */
                namedWindow("diff_people", WINDOW_AUTOSIZE); //WINDOW_OPENGL);
                namedWindow("diff_people2", WINDOW_AUTOSIZE); //WINDOW_OPENGL);
                //  namedWindow("diff_people3", WINDOW_OPENGL);
                imshow("diff_people", gray_people);
                threshold(gray_people, gray_people, 50, 255, THRESH_BINARY);//二值化通常設置為50  255
                Mat element = getStructuringElement(MORPH_RECT, Size(8, 8));
                erode(gray_people, gray_people, element); //scale dwon
              // erode(gray_people, gray_people, element);
                //
               // Mat element = getStructuringElement(MORPH_RECT, Size(15, 10));
              //  dilate(Image_mask, Image_mask, element);
                //
                imshow("diff_people2", gray_people);
                /*
                if (frame_no == 62) {
                    cout << "debig" << "\n";
                }
                */
                //to find people
                vector<vector<Point>> contour_p;
                vector<Vec4i> hierarchy_p;
                findContours(gray_people, contour_p, hierarchy_p, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                grayImage_init_post.release();
                grayImage_now_post.release();
                gray_people.release();

                people_number = 0;
                int people_area = 0;
                int index_p0 = 0;
                int index_p1 = 0;
                int index_p2 = 0;
                int index_p3 = 0;
                int index_p4 = 0;
                int total_p = 0;
                for (int i = 0; i < contour_p.size(); i++) {
                    people_area = (int)contourArea(contour_p.at(i));
                    if (people_area > 700) {
                        //    people_number = people_number + 3;
                        total_p = total_p + 1;
                        if (index_p0 == 0)  index_p0 = i;
                        else if (index_p1 == 0) index_p1 = i;
                        else if (index_p2 == 0) index_p2 = i;
                        else if (index_p3 == 0) index_p3 = i;
                        else if (index_p4 == 0) index_p4 = i;
                        //  printf("the tree area is %d and area is %d\n", i, people_area);
                    }
                    else if (people_area > 400) {
                        //     people_number = people_number + 2;
                        total_p = total_p + 1;
                        if (index_p0 == 0) index_p0 = i;
                        else if (index_p1 == 0) index_p1 = i;
                        else if (index_p2 == 0) index_p2 = i;
                        else if (index_p3 == 0) index_p3 = i;
                        else if (index_p4 == 0) index_p4 = i;
                        //  printf("the two area is %d and area is %d\n", i, people_area);
                    }
                    else if (people_area > 30) {
                        //    people_number = people_number + 1;
                        total_p = total_p + 1;
                        if (index_p0 == 0) index_p0 = i;
                        else if (index_p1 == 0) index_p1 = i;
                        else if (index_p2 == 0) index_p2 = i;
                        else if (index_p3 == 0) index_p3 = i;
                        else if (index_p4 == 0) index_p4 = i;
                    }
                }
                //hugo_diff
               // printf("The frame_no %d peoples %d number %d\n", frame_no, people_number, total_p);

                int p_index = 0;
                if (total_p != 0) {
                    while ((total_p != p_index)) {
                        int index_p_tmp = 0;
                        int tmp_p_start_x = 0, tmp_p_start_y = 0, tmp_p_end_x = 0, tmp_p_end_y = 0;
                        if (p_index == 0) index_p_tmp = index_p0;
                        else if (p_index == 1) index_p_tmp = index_p1;
                        else if (p_index == 2) index_p_tmp = index_p2;
                        else if (p_index == 3) index_p_tmp = index_p3;
                        else if (p_index == 4) index_p_tmp = index_p4;
                        Rect bounding_area_people;
                        bounding_area_people = boundingRect(contour_p[index_p_tmp]);
                        tmp_p_start_x = bounding_area_people.x;
                        tmp_p_start_y = bounding_area_people.y;
                        tmp_p_end_x = tmp_p_start_x + (bounding_area_people.width);
                        tmp_p_end_y = tmp_p_start_y + (bounding_area_people.height);
                        rectangle(C_resize_img_ir, Point((tmp_p_start_x * 4), (tmp_p_start_y * 4)),
                            Point((tmp_p_end_x * 4), (tmp_p_end_y * 4)), Scalar(0, 0, 255), 1, 1, 0);
                        p_index = p_index + 1;
                    }
                }
#endif
                ////
                //printf("The peoples is %d\n",

                if (((abs(pre_x - now_x) < 17) && (abs(pre_y - now_y) < 17) && (abs(tmpNowMin - tmpMin) < 9) && (people_height > 99)
                    && !((pre_x == now_x) && (pre_y == now_y))) || (
                        ((pre_x == now_x) && (pre_y == now_y) && (keep_x == now_x) && (keep_y == now_y))
                        ))
                {
#ifndef net_udp                     
                    show_people_high = 1;
#endif                    
                    first_change = 1;

                    keep_x = now_x;
                    keep_y = now_y;
                    //   keep_value = people_height;
                    check_point_high = 1;
                    if ((abs(keep_old_x - now_x) > 16) || (abs(keep_old_y - now_y) > 16)) {
                        keep_old_x = 0;
                        keep_old_y = 0;
                        keep_old_value = 0;
                        check_point_high = 2;
                    }
                }
                else if (((abs(pre_x - now_x) > 16) || (abs(pre_y - now_y) > 16)) && first_change) {
                    first_change = 0;
                    if (people_height < 90) {
                        keep_old_x = 0;
                        keep_old_y = 0;
                        keep_old_value = 0;
                        check_point_high = 3;
                    }
                    else {
                        keep_old_x = pre_x;
                        keep_old_y = pre_y;
                        keep_old_value = tmpMin;
                        check_point_high = 4;
                    }
                    if (((abs(pre_x - now_x) > 20) || (abs(pre_y - now_y) > 20)) && (people_height > 120) && (people_height_pre > 120)) {
                        other_people_x = pre_x;
                        other_people_y = pre_y;
                        other_people_value = people_height_pre;
                        // check_point_high = 12;
#ifdef PEOPLE_HIGH
                        printf("***big change frame no is %d (%d,%d) = %d %d\n", frame_no, other_people_x, other_people_y, people_height, people_height_pre);
#endif
                    }
                }
                else {
                    check_point_high = 11;
                    if (first_change) {
                        check_point_high = 9;
                        if ((abs(pre_x - now_x) < 17) && (abs(pre_y - now_y) < 17)) {
                            keep_old_x = now_x;
                            keep_old_y = now_y;
                            keep_old_value = tmpNowMin; // grayImage_now.at<uchar>(now_y, now_x); // tmpMin; grayImage_now
                            check_point_high = 5;
                        }
                        else {
                            keep_old_x = pre_x;
                            keep_old_y = pre_y;
                            keep_old_value = tmpMin;
                            check_point_high = 7;
                        }
                    }

                    else if ((no_bed == 1) && (abs(pre_x - now_x) < 17) && (abs(pre_y - now_y) < 17) &&
                        (((abs(keep_old_x - now_x) > 17) || (abs(keep_old_y - now_y) > 17))
                            && (keep_old_x != 0) && (keep_old_y != 0)) && (now_x > 10) && (now_y > 10)) {
                        keep_old_x = now_x;
                        keep_old_y = now_y;
                        keep_old_value = tmpNowMin; // grayImage_now.at<uchar>(now_y, now_x);
                        check_point_high = 10;
                    }

                    first_change = 0;

                    if ((no_bed == 0) && (people_height < 90) && (keep_old_x < 16)) {
                        keep_old_x = 0;
                        keep_old_y = 0;
                        check_point_high = 8;
                    }
                    if (((abs(pre_x - now_x) > 20) || (abs(pre_y - now_y) > 20)) && (people_height > 140) && (people_height_pre > 140)) {
                        other_people_x = pre_x;
                        other_people_y = pre_y;
                        other_people_value = people_height_pre;
                        check_point_high = 12;
#ifdef PEOPLE_HIGH
                        printf("***big change frame no is %d (%d,%d) = %d %d\n", frame_no, other_people_x, other_people_y, people_height, people_height_pre);
#endif
                    }
                    keep_x = 0;
                    keep_y = 0;
                    //keep_value = 0;
                }
#ifdef PEOPLE_HIGH
                if (frame_no > 2)  printf("***frame no is %d*check_point_high %d Now(%d,%d)=%d Pre(%d,%d)=%d Keep(%d,%d) Old(%d,%d)=%d\n",
                    frame_no, check_point_high, now_x, now_y, people_height, pre_x, pre_y, people_height_pre, keep_x, keep_y, //keep_value,
                    keep_old_x, keep_old_y, tmpNowMin);
#endif
            }
            other_people_now = ((grayImage_now.at<uchar>(other_people_y, other_people_x) - tmpNowMin) << 4) / 10;
            if (frame_no == 0) {
#ifdef CAMERA_MODE
                image_0 = image.clone();
                image_ir_0 = image_ir.clone();
#endif
                grayImage_init = grayImage_now_th.clone();
                grayImage_initX = grayImage_now.clone();
                //    Image_per_org = image.clone();
                pre_x = now_x;
                pre_y = now_y;
                tmpMin = tmpNowMin;
                if (initial_begin) {
                    fall_th = P_area; // 8000; // 4000; //1300 // 1000; // 800;
                    edge_level = EDGE_LEVEL; // 10; // 5;
#ifndef net_udp                    
                    String_tmp = IMAGE_PATH;
                    String_tmp = String_tmp + "gis_320x240.jpg";
                    C_grayImage_XX_rise = imread(String_tmp);
                    if (!C_grayImage_XX_rise.data) {
                        C_grayImage_XX_rise = Mat(240, 320, CV_8UC3, Scalar(0));
                    }
#endif                    
                    initial_begin = 0;
                }

                bed_area = bed_area_init;
                bed_area_pre = bed_area_init;
                // row_high = 0;
                col_high = 0;
                trun_lie_down = 0;
                lie_down_mode = 0;
                turn_up = 0;
                turn_up_pre = 0;
                lie_down_ready = 0;
                bed_edge = 0;
                fall_true = 0;
                //     fall_true_pre = 0;
                fall_hold = 0;
                no_people = 0;
                //       no_people_act = 1;
                have_people = 0;
                inside_bed = 0;
                is_inside_bed = 0;
                lie_down_ready = 0;
                people_height_pre = people_height;
                people_counnt_enable = 0;
                /*
                int pre_fall_up_edge = 3;
                int pre_fall_down_edge = 3;
                int pre_fall_threshold = 20;
                */

                printf("=============\n");
                printf("The initial\n");
                printf("#############\n");

                max_hight_init = max_hight_now;
                printf("The frame 0 max high is %d and min %d\n", max_hight_now, tmpNowMin);
                //2022 Display the real field height now 
                float RealHeight = round((float)max_hight_now * 1.569);
                printf("The Approximate Height is %d now  !!\n ", (int)RealHeight);

#ifdef LOG_FILE
                fprintf(fw_log, "The max high is %d and min %d\n", max_hight_now, tmpNowMin);
#endif
            }
            else if (frame_no == 1) {
#ifdef CAMERA_MODE
                image_1 = image.clone();
                image_ir_1 = image_ir.clone();
#endif
                addWeighted(grayImage_now_th, 0.5, grayImage_init, 0.5, 0.0, grayImage_init);
                addWeighted(grayImage_now, 0.5, grayImage_initX, 0.5, 0.0, grayImage_initX);
                max_hight_init = (max_hight_now + max_hight_init) >> 1;
#ifdef IMAGE_INIT
                namedWindow("image_init", WINDOW_AUTOSIZE); //WINDOW_OPENGL);
                imshow("image_init", grayImage_init);
#ifdef SAVE_JPG
                imwrite(image_name_init, grayImage_init);
#endif
#endif

                printf("The origal high %d min %d and avg high %d\n", max_hight_now, tmpNowMin, max_hight_init);
                for (int i = 0; i < grayImage_now.rows; i++) {
                    for (int j = 0; j < grayImage_now.cols; j++) {
                        if (((grayImage_initX.at<uchar>(i, j) < 80) && (grayImage_initX.at<uchar>(i, j) > 5)) ||
                            (grayImage_initX.at<uchar>(i, j) > 220))
                        {
                            Image_mask.at<uchar>(i, j) = 250;
                        }
                        else {
                            Image_mask.at<uchar>(i, j) = 0;
                        }
                    }
                }

                Mat element = getStructuringElement(MORPH_RECT, Size(15, 10));
                dilate(Image_mask, Image_mask, element);

#ifdef IMAGE_MASK
                namedWindow("mask", WINDOW_AUTOSIZE); //WINDOW_OPENGL);
                imshow("mask", Image_mask);
                //debug
                /*
                namedWindow("initX", WINDOW_OPENGL);
                imshow("initX", grayImage_initX);
                */
#endif
                people_height_pre2 = people_height_pre;
                people_height_pre = people_height;
                tmpMin = tmpNowMin;
            }
            else if (frame_no == 2) {
#ifdef CAMERA_MODE
                image_2 = image.clone();
                image_ir_2 = image_ir.clone();
#endif
                printf("The max high is %d and min %d\n", max_hight_now, tmpNowMin);
                grayImage_pre_pre = grayImage_pre.clone();
                grayImage_pre = grayImage_now_th.clone();
                people_height_pre2 = people_height_pre;
                people_height_pre = people_height;

                pre_x = now_x;
                pre_y = now_y;
                tmpMin = tmpNowMin;

                //-----------------------
          //      Image_per_org = image.clone();
                //detect bed area
                Mat_<uchar>  grayImage_bed = grayImage_now;

                no_bed = 1;
                ////////////////////////////////////////////////////////////////////////////////////////
                                // To detect bed
                                ///////////////////////
                bed_count = 0;
                //debug
                int line_status = 0;
                int line_low_high_level = 30;
                // int bed_high_tmp = 0;
                /*
                max_high_less_240 = (max_hight_now < high_diff_th);
                if (max_high_less_240) line_low_high_level = LINE_AVG_HIGH_240;
                else line_low_high_level = LINE_AVG_HIGH_250;
                */
                if (max_hight_now < high_diff_th + 10) {
                    if (max_hight_now < high_diff_th) line_low_high_level = LINE_AVG_HIGH_240;
                    else line_low_high_level = LINE_AVG_HIGH_250;
                }
                else line_low_high_level = LINE_AVG_HIGH_270;

                /////////// line (row) detect
#ifdef BED_LINE_DEBUG_FILE
                fprintf(fw_row_log, " %s high_diff_th %d %d\n", filename_bed_line_log, high_diff_th, max_hight_now);
                fprintf(fw_row_log, "line, value, bed_count_number, status,   bed_count,  line_low_high_level\n");
#endif

                int row_count0 = 0, row_count1 = 0, row_count2 = 0;
                int row_value0 = 0, row_value1 = 0, row_value2 = 0;
                int bed_high_tmp = 0;
                first_line = 0;
                bed_width = 0;
                for (int i = 0; i < grayImage_now.rows; i++) {
                    line_total[i] = 0;
                    for (int j = 30; j < (grayImage_now.cols - 30); j++) {
                        line_total[i] = line_total[i] + grayImage_bed(i, j);
                    }

                    line_total[i] = (line_total[i]) / 100;

                    if (i == 0) {
                        first_value = line_total[i];
                        row_value0 = first_value;
                        row_count0 = 1;
                        bed_count_number = 0;
                        line_status = 1;
                    }
                    else if ((line_total[i] < line_low_high_level) && (((abs(first_value - line_total[i]) < 2))
                        && (i != (grayImage_now.rows - 1))))
                    {
                        if (bed_count_number == 0) first_line = i - 1;
                        bed_count_number = bed_count_number + 1;
                        if (line_total[i] == row_value0) row_count0++;
                        else if (row_value1 == 0) {
                            row_value1 = line_total[i];
                            row_count1++;
                        }
                        else if (line_total[i] == row_value1) row_count1++;
                        else if (row_value2 == 0) {
                            row_value2 = line_total[i];
                            row_count2++;
                        }
                        else if (line_total[i] == row_value2) row_count2++;
                        line_status = 2;
                        if (line_total[i] > first_value) {
                            first_value = line_total[i]; //temp check
                            line_status = 3;
                        }
                        // if (bed_count_number == 10) bed_high_tmp = line_total[i];
                    }
                    else {
                        line_status = 4;
                        if (bed_count_number > 19) {
                            if (bed_count == 0) {
                                if (first_line == 0) bed_first_line = 0;
                                else bed_first_line = first_line - 1;
                                if (line_total[i] < first_value) {
                                    bed_width = bed_count_number - 1;
                                    sick_house = 1;
                                }
                                else if ((abs(first_value - line_total[i]) < 2)) bed_width = bed_count_number + 2;
                                else bed_width = bed_count_number + 1;
                                bed_end_line = first_line + bed_width;
                                //         bed_half_line = first_line + (bed_width >> 1);
                                if ((row_count0 > row_count1) && (row_count0 > row_count2)) {
                                    bed_high_tmp = ((row_value0 + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
                                else if ((row_count1 > row_count0) && (row_count1 > row_count2)) {
                                    bed_high_tmp = ((row_value1 + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
                                else if ((row_count2 > row_count0) && (row_count2 > row_count1)) {
                                    bed_high_tmp = ((row_value2 + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
                                no_bed = 0;
#ifdef SECOND_METHOD
                                bed_second_method = 0;
#endif
                                bed_count_number = 0;
                                line_status = 5;
#ifdef BED_LINE_DEBUG
                                printf("The first_value %d row_value0 %d sick_house %d\n", first_value, row_value0, sick_house);
                                printf("The1 count0 (%d,%d) count1 (%d,%d) count2 (%d,%d)\n", row_count0, row_value0, row_count1, row_value1, row_count2, row_value2);
#endif
                            }
                            else if (((bed_count_number + 1) >= bed_width) && (bed_high > ((first_value >> 1) * 10))) {//bed_high
                                if (first_line == 0) bed_first_line = 0;
                                else bed_first_line = first_line - 1;
                                if (line_total[i] < first_value) {
                                    bed_width = bed_count_number - 1;
                                    sick_house = 1;
                                }
                                else if ((abs(first_value - line_total[i]) < 2)) bed_width = bed_count_number + 2;
                                else bed_width = bed_count_number + 1;
                                bed_end_line = first_line + bed_width - 1;
                                //            bed_half_line = first_line + (bed_width >> 1);
                                if ((row_count0 > row_count1) && (row_count0 > row_count2)) {
                                    bed_high_tmp = ((first_value + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
                                else if ((row_count1 > row_count0) & (row_count1 > row_count2)) {
                                    bed_high_tmp = ((row_value1 + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
                                else if ((row_count2 > row_count0) & (row_count2 > row_count1)) {
                                    bed_high_tmp = ((row_value2 + 1) >> 1);
                                    bed_high = bed_high_tmp * 10;
                                }
#ifdef SECOND_METHOD
                                bed_second_method = 0;
#endif
                                no_bed = 0;
                                line_status = 6;
#ifdef BED_LINE_DEBUG
                                printf("The2 count0 (%d,%d) count1 (%d,%d) count2 (%d,%d)\n", row_count0, row_value0, row_count1, row_value1, row_count2, row_value2);
#endif
                            }
                            bed_count = bed_count + 1;
                        } // (bed_count_number > 19)
//#ifdef SECOND_METHOD
                        else {

                            first_value = line_total[i];
                            row_value0 = first_value;
                            row_count0 = 1;
                            row_value1 = 0;
                            row_value2 = 0;
                            row_count1 = 0;
                            row_count2 = 0;
                            bed_count_number = 0;
                        }
                        //  bed_high_tmp = 0;
//#endif
                    } //j

#ifdef BED_LINE_DEBUG
                    printf("line %2d is %d and bed_count_number:%d status: %d %d\n", i, line_total[i], bed_count_number, line_status, bed_count);
#endif
#ifdef BED_LINE_DEBUG_FILE
                    fprintf(fw_row_log, " %2d,  %2d, %3d, %2d, %d\n", i, line_total[i], bed_count_number, line_status, bed_count);
#endif

                } //i
#ifdef BED_LINE_DEBUG
                printf("The3 count0 (%d,%d) count1 (%d,%d) count2 (%d,%d)\n", row_count0, row_value0, row_count1, row_value1, row_count2, row_value2);
#ifdef SECOND_METHOD
                printf("The bed is %d and %d\n", no_bed, bed_second_method);
#else
                printf("The no bed is %d \n", no_bed);
#endif
#endif
#ifdef BED_LINE_DEBUG_FILE
                fprintf(fw_row_log, "Row first %d end %d and width %d hight %d line_low_high_level%d\n", bed_first_line, bed_end_line, bed_width, bed_high, line_low_high_level);
                fclose(fw_row_log);
#endif
                //////////////////////////////////////////////////////
#ifdef SECOND_METHOD
                if (bed_second_method) {
#ifdef BED_LINE_DEBUG
                    printf("The secind method row\n");
#endif
                    int i = 0;
                    bool up_is1 = 0;
                    bool up_is2 = 0;
                    bool up_is3 = 0;
                    int bed_high_up = 0;
                    int bed_high_down = 0;
                    int bed_high_up2 = 0;
                    int bed_high_up3 = 0;
                    int bed_high_down2 = 0;
                    int bed_high_down3 = 0;
                    int init_line_begin = 20;
                    int init_line_end = (grayImage_now.cols - 40);

                    do {
                        line_total[i] = 0;
                        for (int j = init_line_begin; j < init_line_end; j++) {
                            line_total[i] = line_total[i] + grayImage_bed(i, j);
                        }
                        line_total[i] = (line_total[i]) / 100;
                        if (i == 0) {
                            first_value = line_total[i];
                        }
                        else if ((!up_is1) && (abs(first_value - line_total[i]) > 4)) {
                            up_is1 = 1;
                            bed_first_line = i;
                            bed_high_up = (line_total[i] >> 1) * 10;
                            // printf("up_is1 %d, %d first %d \n", up_is1, i, first_value);
                        }
                        else if (up_is1 && !up_is2) {
                            up_is1 = 1;
                            up_is2 = 1;
                            bed_high_up2 = (line_total[i] >> 1) * 10;
                            //debug
                           // printf("get %d bed_first_line %d and upis1 %d \n", i, bed_first_line, up_is1);
                        }
                        else if (up_is2 && !up_is3) {
                            up_is2 = 1;
                            up_is3 = 1;
                            bed_high_up3 = (line_total[i] >> 1) * 10;
                            //debug
                           // printf("get %d bed_first_line %d and upis1 %d \n", i, bed_first_line, up_is1);
                        }

                        i++;
                    } while ((!up_is3) & (i < 40));
#ifdef BED_LINE_DEBUG
                    printf("The up bed high %d %d %d %d\n", i, bed_high_up, bed_high_up2, bed_high_up2);
#endif
                    i = 59;
                    bool down_is1 = 0;
                    bool down_is2 = 0;
                    bool down_is3 = 0;
                    do {
                        line_total[i] = 0;
                        for (int j = init_line_begin; j < init_line_end; j++) {
                            line_total[i] = line_total[i] + grayImage_bed(i, j);
                        }
                        line_total[i] = (line_total[i]) / 100;

#ifdef BED_LINE_DEBUG
                        printf("The down line %d is %d \n", i, line_total[i]);
#endif

                        if (i == 59) first_value = line_total[i];
                        else if ((!down_is1) && ((abs(first_value - line_total[i]) > 4))) {
                            down_is1 = 1;
                            bed_end_line = i;
                            bed_high_down = (line_total[i] >> 1) * 10;
                        }
                        else if (down_is1 && !down_is2) {
                            down_is1 = 1;
                            down_is2 = 1;
                            bed_high_down2 = (line_total[i] >> 1) * 10;
                        }
                        else if (down_is2 && !down_is3) {
                            down_is2 = 1;
                            down_is3 = 1;
                            bed_high_down3 = (line_total[i] >> 1) * 10;
                        }
                        i--;
                    } while ((!down_is3) & (i > 10));
#ifdef BED_LINE_DEBUG
                    printf("The down bed high %d %d %d %d\n", bed_end_line, bed_high_down, bed_high_down2, bed_high_down3);
#endif
                    int count_up = 0;
                    if (bed_high_up == bed_high_down)    count_up++;
                    if (bed_high_up == bed_high_down2)   count_up++;
                    if (bed_high_up == bed_high_down3)   count_up++;

                    int count_up2 = 0;
                    if (bed_high_up2 == bed_high_down)    count_up2++;
                    if (bed_high_up2 == bed_high_down2)   count_up2++;
                    if (bed_high_up2 == bed_high_down3)   count_up2++;

                    int count_up3 = 0;
                    if (bed_high_up3 == bed_high_down)    count_up3++;
                    if (bed_high_up3 == bed_high_down2)   count_up3++;
                    if (bed_high_up3 == bed_high_down3)   count_up3++;
                    int count_up4 = 0;
                    if (bed_high_up == bed_high_up2)    count_up4++;
                    if (bed_high_up == bed_high_up3)    count_up4++;
                    int count_up5 = 0;
                    if (bed_high_down == bed_high_down2)    count_up5++;
                    if (bed_high_down == bed_high_down3)    count_up5++;
#ifdef BED_LINE_DEBUG
                    printf("1x:%d, 2x:%d, 3x:%d 4x:%d 5x:%d\n", count_up, count_up2, count_up3, count_up4, count_up5);
#endif
                    bed_high = 0;
                    no_bed = 0;
                    int tmp_count0 = 0;
                    int tmp_bed_high0 = 0;
                    if (count_up > count_up2) {
                        tmp_count0 = count_up;
                        tmp_bed_high0 = bed_high_up;
                    }
                    else {
                        tmp_count0 = count_up2;
                        tmp_bed_high0 = bed_high_up2;
                    }
                    int tmp_count1 = 0;
                    int tmp_bed_high1 = 0;
                    if (count_up3 > count_up4) {
                        tmp_count1 = count_up3;
                        tmp_bed_high1 = bed_high_up3;
                    }
                    else {
                        tmp_count1 = count_up4;
                        tmp_bed_high1 = bed_high_up;
                    }
#ifdef BED_LINE_DEBUG
                    printf("The tmp_count1 %d tmp_bed_high1 %d \n", tmp_count1, tmp_bed_high1);
#endif
                    ///
                    int tmp_count2 = 0;
                    int tmp_bed_high2 = 0;
                    if (tmp_count0 > tmp_count1) {
                        tmp_count2 = tmp_count0;
                        tmp_bed_high2 = tmp_bed_high0;
                    }
                    else {
                        tmp_count2 = tmp_count1;
                        tmp_bed_high2 = tmp_bed_high1;

                    }
#ifdef BED_LINE_DEBUG
                    printf("The tmp0 %d tmp 1 %d and tmp2 %d count_up5 %d\n", tmp_count0, tmp_count1, tmp_count2, count_up5);
#endif
                    if (((tmp_count2 == 0) && (count_up5 == 0)) || (tmp_count2 == count_up5)) no_bed = 1;
                    else if (tmp_count2 > count_up5) {
                        bed_high = tmp_bed_high2;
                    }
                    else {
                        bed_high = bed_high_down;
                    }

                    bed_width = 0;
                    if (no_bed == 0) {
                        //if (bed_end_line < bed_first_line) bed_end_line = 59;
                        bed_width = (bed_end_line - bed_first_line) + 1;
                        //           bed_half_line = bed_first_line + (bed_width >> 1);
                    }
#ifdef BED_LINE_DEBUG
                    printf("The bed_width %d, end %d first %d\n", bed_width, bed_end_line, bed_first_line);
#endif
                    if (bed_width < 15) {
                        no_bed = 1;
#ifdef BED_LINE_DEBUG
                        printf("NO Bed and width %d and hight %d\n", bed_width, bed_high);
#endif
                    }
                    else {
                        no_bed = 0;
#ifdef BED_LINE_DEBUG
                        printf("have bed and width %d and hight %d\n", bed_width, bed_high);
#endif
                    }
                } //bed_second_method
#endif
#ifdef LOG_FILE
                if (no_bed == 0) {
                    if (bed_second_method) fprintf(fw_log, "The 2rd  row first %d, end %d width %d and high %d\n", bed_first_line, bed_end_line, bed_width, bed_high);
                    else fprintf(fw_log, "Row first %d end %d and width %d hight %d\n", bed_first_line, bed_end_line, bed_width, bed_high);
                }
                else fprintf(fw_log, " NO bed\n");
#endif
                printf("Row first %d end %d and width %d hight %d\n", bed_first_line, bed_end_line, bed_width, bed_high);
#ifdef SECOND_METHOD

                if (no_bed == 0) {
                    if (bed_second_method) printf("The 2rd  row first %d, end %d width %d and high %d\n", bed_first_line, bed_end_line, bed_width, bed_high);
                    else printf("Row first %d end %d and width %d hight %d\n", bed_first_line, bed_end_line, bed_width, bed_high);
                    int down_level_edge = (60 - bed_width) >> 1;
                    int down_level_edge_up = down_level_edge - 7;
                    int down_level_edge_down = down_level_edge + 7;
                    bed_position_up = 0;
                    bed_position_down = 0;
                    pre_fall_up_edge = 3;
                    pre_fall_down_edge = 3;
                    if (bed_width < 27) {
                        if (bed_first_line < down_level_edge_up) {
                            edge_up_limit = 0;
                            bed_position_up = 1;
                            pre_fall_up_edge = 1;
                            pre_fall_down_edge = 2;
                        }
                        else edge_up_limit = 2;
                        if (bed_first_line > down_level_edge_down) {
                            edge_down_limit = 0;
                            bed_position_down = 1;
                            pre_fall_up_edge = 2;
                            pre_fall_down_edge = 1;
                        }
                        else edge_down_limit = 4;
                    }
                    else if (bed_width < WIDTH_TH) {
                        if (bed_first_line < down_level_edge_up) {
                            edge_up_limit = 0;
                            pre_fall_up_edge = 1;
                            pre_fall_down_edge = 2;
                            bed_position_up = 1;
                        }
                        else edge_up_limit = 3;
                        if (bed_first_line > down_level_edge_down) {
                            edge_down_limit = 0;
                            bed_position_down = 1;
                            pre_fall_up_edge = 2;
                            pre_fall_down_edge = 1;
                        }
                        else edge_down_limit = 4;
                    }
                    else {
                        if (bed_first_line < down_level_edge_up) {
                            edge_up_limit = 2;
                            bed_position_up = 1;
                            pre_fall_up_edge = 1;
                            pre_fall_down_edge = 2;
                        }
                        else edge_up_limit = 5;
                        if (bed_first_line > down_level_edge_down) {
                            edge_down_limit = 4;
                            bed_position_down = 1;
                            pre_fall_up_edge = 2;
                            pre_fall_down_edge = 1;
                        }
                        else edge_down_limit = 6;
                    }
                    //bed_position_center = !(bed_position_up | bed_position_down);
                    bed_height = ((max_hight_now - bed_high) * 16) / 10;

                } //if(!no_bed

#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                if (no_bed == 0) {
                    //col_total
#ifdef SECOND_METHOD
       //             if (!bed_second_method) {
#endif
                        //   printf("The first col method\n");
#ifdef BED_COL_DEBUG_FILE
                    fprintf(fw_col_log, " %s max_hight_now %d\n", filename_bed_col_log, max_hight_now);
                    fprintf(fw_col_log, "col, value, bed_count, status\n");
#endif
                    bed_count_number = 0;
                    bed_count = 0;

                    int col_ini_status = 0;
                    int first_value_pos = 0;
                    for (int j = 0; j < grayImage_now.cols; j++) {
                        col_total_init[j] = 0;
                        for (int i = (bed_first_line); i < (bed_end_line); i++) {
                            col_total_init[j] = col_total_init[j] + grayImage_bed(i, j);
                        }

                        int col_total_tmp = (col_total_init[j] + 50);
                        col_total_init[j] = (col_total_tmp / 100);
                        //                           col_total_init[j] = col_total_init[j]/bed_width;
                        if (j == 0) {
                            first_value = col_total_init[j];
                            // bed_col_first_count = 1;
                            bed_count_number = 0;
                            col_ini_status = 1;
                        }
                        else if ((j == 1) && (abs(first_value - col_total_init[j]) < 2) && (j != (grayImage_now.cols - 1)))
                        {
                            if (bed_count_number == 0) first_line = j - 1;
                            bed_count_number = bed_count_number + 1;
                            col_ini_status = 2;
                        }
                        else if (((first_value_pos + 1) == j) & (first_value == col_total_init[j])) {
                            first_line = j;
                            bed_count_number = 1;
                        }
                        else if ((abs(col_total_init[j - 1] - col_total_init[j]) < 2) && (abs(first_value - col_total_init[j]) < 3) &&
                            (abs(col_total_init[j] - col_total_init[j - 2]) < 2) && (j != (grayImage_now.cols - 1))
                            && (bed_count_number != 0))
                        { // (abs(col_total_init[j] - col_total_init[j-2]) < 3)
//                            if (bed_count_number == 0) first_line = j;
                            bed_count_number = bed_count_number + 1;
                            col_ini_status = 3;
                        }
                        else if (bed_count_number > 16)
                        {
                            col_ini_status = 4;
                            if (bed_count == 0) {
                                col_first_line = first_line;
                                col_width = bed_count_number + 1;
                                col_end_line = first_line + col_width;
                                col_high = first_value;
                                col_ini_status = 5;
                            }
                            else if (col_width < (bed_count_number + 1)) {
                                col_first_line = first_line;
                                col_width = bed_count_number + 1;
                                col_end_line = first_line + col_width;
                                col_high = first_value;
                                col_ini_status = 6;
                            }
                            bed_count = bed_count + 1;
                            bed_count_number = 0;
                            first_value = col_total_init[j];
                        }
                        else {
                            first_value = col_total_init[j];
                            first_value_pos = j;
                            bed_count_number = 0;
                            col_ini_status = 7;
                        }
#ifdef BED_COL_DEBUG
                        printf("The col %2d is %d and %d status %d\n", j, col_total_init[j], bed_count_number, col_ini_status);
#endif
#ifdef BED_COL_DEBUG_FILE
                        fprintf(fw_col_log, "%2d,  %d, %d,  %d\n", j, col_total_init[j], bed_count_number, col_ini_status);
#endif
                    }
#ifdef BED_COL_DEBUG_FILE
                    fprintf(fw_col_log, "count %d col first %d end %d\n", bed_count, col_first_line, col_end_line);
                    fprintf(fw_col_log, "width %d\n", col_width);
                    fprintf(fw_col_log, "high %d\n", col_high);
                    fclose(fw_col_log);
#endif
#ifdef SECOND_METHOD
                    //       }
#endif
#ifdef SECOND_METHOD
                else {
#ifdef BED_COL_DEBUG
                    printf("The second col method\n");
#endif
                    // col_total_init[j] = 0;
                    bool first_col = 1, end_col_is = 1;
                    int col_diff = 0;
                    int j = 0;
                    int col_begin_over = 0;
                    int col_end_over = 0;
                    //                      for (int j = 0; j < (grayImage_now.cols); j++) {
                    do {
                        col_diff = abs(grayImage_bed((bed_first_line + 2), j) - bed_high);
                        if ((col_diff < 3) && first_col) {
                            col_first_line = j + 1;
                            first_col = 0;
                        }
                        j++;
                        col_begin_over = (j == (grayImage_now.cols - 1));
#ifdef BED_COL_DEBUG
                        cout << "the j: " << j << " diff " << col_diff << endl;
#endif
                    } while ((first_col) && !col_begin_over);
                    if (col_begin_over) no_bed = 1;
                    else {
                        j = (grayImage_now.cols - 2);
                        do {
                            col_diff = abs(grayImage_bed((bed_first_line + 2), j) - bed_high);
                            if ((col_diff < 3) && (end_col_is)) {
                                col_end_line = j - 1;
                                end_col_is = 0;
                            }
                            j--;
                            col_end_over = (j == 1);
#ifdef BED_COL_DEBUG
                            cout << "last j: " << j << " diff " << col_diff << endl;
#endif
                        } while (end_col_is & (!col_end_over));
                    }
#ifdef LOG_FILE
                    fprintf(fw_log, "The col first %d, end %d width %d and  high %d\n", col_first_line, col_end_line, col_width, col_high);
#else
                    cout << "the j: " << j << " diff " << col_diff << " first " << col_first_line << " end " << col_end_line << endl;
#endif


                    if (col_end_line < col_first_line) no_bed = 1;
                    else {
                        col_width = col_end_line - col_first_line + 1;
                        for (int j = col_first_line; j < (col_end_line + 1); j++)
                        {
                            col_total_init[j] = 0;
                            for (int i = bed_first_line; i < bed_end_line; i++)
                            {
                                col_total_init[j] = col_total_init[j] + bed_high;
                            }
                            int col_tatal_tmp = (col_total_init[j] + 50);
                            col_total_init[j] = (col_tatal_tmp / 100);
                        }
                    }
                    col_high = col_total_init[col_first_line];
#ifdef BED_COL_DEBUG
                    printf("The second medtho col first %d, end %d width %d and  high %d\n", col_first_line, col_end_line, col_width, col_high);
#endif

                }
#endif
                //if (col_high < 15) no_bed = 1;
                printf("The col result first line %d and width %d and last %d and depth %d\n", col_first_line
                    , col_width, col_end_line, col_high);
                /*
                                if ((bed_width > (WIDTH_TH + 15)) || (bed_width < WIDTH_TH) || (col_width < BED_LENGTH) || (col_high < 15)) no_bed = 1;
                                else no_bed = 0;
                                */
                                //高度在2.4 ~ 2.5 
                if (max_hight_now < high_diff_th + 10) {
                    if ((bed_width > (WIDTH_TH + 15)) || (bed_width < WIDTH_TH) || (col_width < BED_LENGTH) || (col_high < 15)) no_bed = 1;
                    else no_bed = 0;
                }
                //高度在2.5 ~ 以上

                else {
                    if ((bed_width > (WIDTH_TH - 5 + 15)) || (bed_width < WIDTH_TH - 5) || (col_width < BED_LENGTH) || (col_high < 15)) no_bed = 1;
                    else no_bed = 0;
                }
#ifdef Wayne_debug
                printf("\n max_hight_now : %d", max_hight_now);
                printf("\n high_diff_th : %d", high_diff_th);
                printf("\n  bed_width : %d", bed_width);
                printf("\n  WIDTH_TH: %d", WIDTH_TH);
                printf("\n  BED_LENGTH: %d", BED_LENGTH);
                printf("\n  col_width : %d", col_width);
                printf("\n  col_high : %d", col_high);
#endif
                cout << "The bed width " << bed_width << "The col length " << col_width << " bed high " << col_high << endl;


                //  no_bed = 1;
#ifdef LOG_FILE
                if (no_bed == 0) {
                    if (bed_second_method) fprintf(fw_log, "The second method col first %d, end %d width %d and  high %d\n", col_first_line, col_end_line, col_width, col_high);
                    else fprintf(fw_log, "The normal method col first %d, end %d width %d and high %d\n", col_first_line, col_end_line, col_width, col_high);
                }
#else
                if (no_bed == 0) {
#ifdef SECOND_METHOD
                    if (bed_second_method) {
                        printf("The second method col first %d, end %d width %d and  high %d\n", col_first_line, col_end_line, col_width, col_high);
                        if (display_bed_detect) sprintf(tmp_status, "LieDown");
                        lie_down_ready = 1;
                        is_lie_down = 1;
                        trun_lie_down = 1;
                        real_bed_mode = 1;
                    }
                    else
#endif
                        printf("The normal method col first %d, end %d width %d and high %d\n", col_first_line, col_end_line, col_width, col_high);
                    // edge_level = 7;
                }
                else {
#ifdef DEBUG_PRINT                    
                    printf("can not find bed\n");
#endif                    
                    bed_area = bed_area_init;
                }
#endif
                ///////////////////////////////////////////////
#ifdef SECOND_METHOD
                if (no_bed == 0) printf("have bed and %d\n", bed_second_method);
                else {
                    printf("can not find bed\n");
                    bed_area = bed_area_init;
                }
#endif
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                } //if(!no_bed)
////////////////////////////////////////////////////////////////////////////////////////
#ifndef net_udp
#ifndef IP_CAMERA
                namedWindow("controller", WINDOW_AUTOSIZE); //WINDOW_NORMAL);
                if (no_bed == 1)resizeWindow("controller", 400, 50);
                else  resizeWindow("controller", 400, 100);
#ifdef HUGO_DEBUG
                cvCreateTrackbar("抽圖", "controller", &draw_frame, 10, NULL);
#endif

                if (no_bed == 0) {
                    //  createTrackbar("預跌高度面積值", "controller", &pre_fall_threshold, 50, NULL);
                   //  createTrackbar("預跌上線", "controller", &pre_fall_up_edge, 10, NULL);
                    // createTrackbar("預跌下線", "controller", &pre_fall_down_edge, 10, NULL);
 //                    createTrackbar("開啟預跌", "controller", &is_pre_fall, 1, NULL);
 //                    createTrackbar("秀預跌", "controller", &show_pre_fall, 1, NULL);
                    // createTrackbar("關左右翻身", "controller", &no_check_left_right, 1, NULL);
                    createTrackbar("無床", "controller", &no_bed, 1, NULL);
                }
                else {
                    createTrackbar("無床", "controller", &no_bed, 1, NULL);
                }
                createTrackbar("秀幀數", "controller", &frame_no_is, 1, NULL);

#ifdef HUGO_DEBUG
                cvCreateTrackbar("秀床數值", "controller", &display_bed_detect, 1, NULL);
                cvCreateTrackbar("秀跌數值", "controller", &show_fall_message, 1, NULL);
                cvCreateTrackbar("秀跌pre數值", "controller", &show_fall_message_pre, 1, NULL);
#endif
#endif
#endif
                // 2022 抓錯
                if (no_bed == 0) {
                    for (int i = bed_first_line; i < bed_end_line; i++) {
                        line_total_init[i] = 0;
                        line_half_init[i] = 0;
                        for (int j = col_first_line; j < col_end_line; j++) {
                            line_total_init[i] = line_total_init[i] + grayImage_bed(i, j);
                            if (j < (col_end_line >> 1))
                                line_half_init[i] = line_half_init[i] + grayImage_bed(i, j);
                        }
                        int line_totoal_tmp1 = (line_total_init[i] + 50);
                        int line_half_tmp1 = (line_half_init[i] + 50);
                        line_total_init[i] = line_totoal_tmp1 / 100;
                        line_half_init[i] = line_half_tmp1 / 50;
                        if (i == 0) {
                            first_value = line_total_init[i];
                            bed_count_number = 0;
                        }
                        else if ((abs(first_value - line_total_init[i]) < 2) & (i != (bed_end_line - 1)))
                        {
                            bed_count_number = bed_count_number + 1;
                        }
                        else if (bed_count_number > 10)
                        {
                            row_high = first_value;
                        }
                        else {
                            first_value = line_total_init[i];
                            bed_count_number = 0;
                        }
                        //#ifdef DEBUG_NEW
                         //                           printf("The row %2d is %d and %d\n", i, line_total_init[i], bed_count_number);
                        //#endif
                    }
                    // ### 2022 Revised #####
                        //Up
                    int PreOutSideUpArea = 0;
                    for (int i = 0; i <= bed_first_line; i++) {
                        for (int j = col_first_line; j < col_end_line; j++) {
                            PreOutSideUpArea = PreOutSideUpArea + 1;
                            GroundAvgUpPixel = GroundAvgUpPixel + now_image(i, j);
                        }
                    }
                    if (PreOutSideUpArea != 0)GroundAvgUpPixel = GroundAvgUpPixel / PreOutSideUpArea;
                    //Down 
                    int PreOutSideDownArea = 0;
                    for (int i = bed_end_line + 1; i < 59; i++) {
                        for (int j = col_first_line; j < col_end_line; j++) {
                            PreOutSideDownArea = PreOutSideDownArea + 1;
                            GroundAvgDownPixel = GroundAvgDownPixel + now_image(i, j);
                        }
                    }
                    if (PreOutSideDownArea != 0) GroundAvgDownPixel = GroundAvgDownPixel / PreOutSideDownArea;
                    // ### 2022 Revised End #####
                    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                }//!no_beds
            } //frame_no == 2

            //initial end
            else {
                // cout << frame_no << "  " << init_data << " \n";
                ////////////////////////////////On the bed///////////////////////////////////////////////////////////////////////////////////
                //To find the people
                if (no_bed == 0) {
                    //detect bed area
                    bool first_row_mark = true;
                    int first_row_position = 0;
                    int bed_row_low_count = 0;
                    int bed_row_low_count2 = 0;
                    int row_diff = 0;
                    int zero_count = 0;
                    bed_area.start_y = 0;
                    bed_area.end_y = 0;
                    int  bed_row_low_count_tmp = 0;
                    int bed_area_star_y_tmp = 0;
                    Mat_<uchar>  grayImage_bed = grayImage_now;
                    int ChangePixel = 0; //20220126
                    int countPixel = 0; //20220126

                    for (int i = bed_first_line; i < bed_end_line; i++) {
                        int status_row = 0;
                        line_total[i] = 0;
                        for (int j = col_first_line; j < col_end_line; j++) {
                            line_total[i] = line_total[i] + grayImage_bed(i, j);
                        }

                        int line_total_tmp = (line_total[i] + 50);
                        line_total[i] = line_total_tmp / 100;
                        if (line_total_init[i] > line_total[i]) row_diff = line_total_init[i] - line_total[i];
                        else row_diff = 0;

                        if ((row_diff > 1) & (i < (bed_end_line - 1))) {
                            zero_count = 0;
                            if (first_row_mark) {

                                bed_area_star_y_tmp = i;
                                first_row_mark = false;
                                bed_row_low_count++;

                                status_row = 1;

                                first_row_position = i;
                            }
                            else if (i == (first_row_position + 1)) {
                                first_row_position = i;
                                status_row = 2;
                                bed_row_low_count++;
                                /*
                                if (row_diff > 2) {
                                    if (bed_row_low_count2 == 0) second_starty = i;
                                    bed_row_low_count2++;
                                }
                                */
                            }
                        }
                        else if ((row_diff < 2) & (zero_count < 1) & (i != (bed_end_line - 1))) {
                            zero_count = zero_count + 1;
                            status_row = 3;
                            first_row_position = i;
                            bed_row_low_count++;
                        }
                        else {
                            if (i == (first_row_position + 1)) {
                                if ((bed_row_low_count > 5) & (zero_count < 2) & (bed_row_low_count_tmp < bed_row_low_count)) {
                                    bed_row_low_count_tmp = bed_row_low_count;
                                    status_row = 4;
                                    bed_area.start_y = bed_area_star_y_tmp;
                                    if (bed_area.start_y == 0) bed_area.start_y = 1;
                                    bed_area.end_y = i;
                                    bed_area.width_y = bed_row_low_count;
                                    bed_area.center_y = bed_area.start_y + ((bed_area.width_y + 1) >> 1);
                                    bed_area.before_y = bed_area.start_y + ((bed_area.width_y + 2) >> 2);

                                }
                                else if ((bed_row_low_count < 7) || (zero_count > 0)) {
                                    first_row_mark = true;
                                    first_row_position = i;
                                    status_row = 5;
                                }
                                if (i < (bed_end_line - 1)) first_row_mark = true;
                                else first_row_mark = false;
                                zero_count = 0;
                                bed_row_low_count = 0;
                                first_row_position = i;
                                bed_row_low_count2 = 0;
                            }
                        }

                    }
                    //////////////////////////////////////////////////////////////////////////////////////////////////////
                    bed_row_low_count = 0;

                    bed_row_low_count_tmp = 0;
                    first_row_mark = true;
                    first_row_position = 0;
                    bed_area_star_y_tmp = 0;
                    zero_count = 0;
                    bed_area.start2_y = bed_area.start_y;
                    bed_area.end2_y = bed_area.end_y;
                    for (int i = bed_first_line; i < bed_end_line; i++) {
                        int status_half = 0;
                        line_half[i] = 0;
                        for (int j = col_first_line; j < col_end_line - 4; j++) { //20220126
                            if (j < (col_end_line >> 1))  line_half[i] = line_half[i] + grayImage_bed(i, j);
                            //20220126 start
                            int ChangePixel = abs(grayImage_bed(i, j) - grayImage_pre.at<uchar>(i, j));
                            if (ChangePixel > 10) {
                                countPixel = countPixel + 1;
                            }
                            //20220126 end
                        }
                        int line_half_tmp = (line_half[i] + 50);
                        line_half[i] = line_half_tmp / 50;
                        if (line_half_init[i] > line_half[i]) row_diff = line_half_init[i] - line_half[i];
                        else row_diff = 0;

                        int diff_value = 1;
                        ///
                        if ((row_diff > diff_value) & (i < (bed_end_line - 1))) {
                            zero_count = 0;
                            status_half = 1;
                            if (first_row_mark) {
                                bed_area_star_y_tmp = i;
                                first_row_mark = false;
                                first_row_position = i;
                                bed_row_low_count++;
                                status_half = 2;
                            }
                            else if (i == (first_row_position + 1)) {
                                first_row_position = i;
                                bed_row_low_count++;
                                status_half = 3;
                            }
                        }
                        else if ((row_diff <= diff_value) & (zero_count < 1) & (i != (bed_end_line - 1))) {
                            zero_count = zero_count + 1;
                            first_row_position = i;
                            status_half = 4;
                            // bed_col_low_count++;
                        }
                        else {
                            if (i == (first_row_position + 1)) {
                                if ((bed_row_low_count > 4) && (zero_count < 2) && (bed_row_low_count_tmp < bed_row_low_count)) {
                                    bed_row_low_count_tmp = bed_row_low_count;
                                    bed_area.start2_y = bed_area_star_y_tmp;
                                    bed_area.end2_y = i;
                                    bed_area.width2_y = bed_row_low_count;
                                    first_row_mark = true;
                                    first_row_position = i;
                                    bed_row_low_count = 0;
                                    zero_count = 0;
                                    status_half = 5;
                                }
                                else if ((bed_row_low_count < 7) || (zero_count > 0)) {
                                    first_row_mark = true;
                                    first_row_position = i;
                                    bed_row_low_count = 0;
                                    zero_count = 0;
                                    status_half = 6;
                                }

                            }
                        }
#ifdef DEBUG_BED_ROW2
                        if (frame_no > 966)
                            printf("the 2rd %d status %d, row_diff %d count %d line_half %d line %d \n", i, status_half, row_diff, bed_row_low_count,
                                line_half_init[i], line_half[i]);
#endif
                    }
                    if (countPixel >= 15) bed_obj_status = true; //20220126
                    else bed_obj_status = false; //20220126


                    // if (bed_area.start2_y)
                    // printf("The starty %d endy %d start2y %d endy2 %d\n", bed_area.start_y, bed_area.end_y,bed_area.start2_y, bed_area.end2_y);
                    //col_total
                    bool first_col_mark = true;
                    int first_col_position = 0;
                    int bed_col_low_count = 0;
                    int  bed_col_low_count_tmp = 0;
                    int col_diff = 0;
                    zero_count = 0;
                    bed_area.start_x = 0;
                    bed_area.end_x = 0;
                    int bed_area_star_x_tmp = 0;

                    int people_col_status = 0;
                    for (int j = col_first_line; j < col_end_line; j++) {
                        col_total[j] = 0;
                        for (int i = bed_first_line; i < bed_end_line; i++) {
                            col_total[j] = col_total[j] + grayImage_bed(i, j);
                        }
                        int col_total_tmp = (col_total[j] + 50);
                        col_total[j] = col_total_tmp / 100;
                        if (col_total_init[j] > col_total[j]) col_diff = col_total_init[j] - col_total[j];
                        else col_diff = 0;
                        if ((col_diff > 0) && (j < (col_end_line - 1))) {
                            zero_count = 0;
                            people_col_status = 1;
                            if (first_col_mark) {
                                bed_area_star_x_tmp = j;
                                first_col_mark = false;
                                first_col_position = j;
                                bed_col_low_count++;
                                people_col_status = 2;
                            }
                            else if (j == (first_col_position + 1)) {
                                first_col_position = j;
                                bed_col_low_count++;
                                people_col_status = 3;
                            }
                        }
                        else if ((col_diff == 0) && (zero_count < 1) && (j != (col_end_line - 1))) {
                            zero_count = zero_count + 1;
                            first_col_position = j;
                            bed_col_low_count++;
                            people_col_status = 4;
                        }
                        else {
                            if ((bed_col_low_count > 6) && (zero_count < 2) && (bed_col_low_count_tmp < bed_col_low_count)) {
                                bed_col_low_count_tmp = bed_col_low_count;
                                bed_area.start_x = bed_area_star_x_tmp;
                                if (bed_area.start_x == 0) bed_area.start_x = 1;
                                if (zero_count == 1) {
                                    bed_area.end_x = j - 2;
                                    bed_area.width_x = bed_col_low_count - 2;
                                }
                                else {
                                    bed_area.end_x = j - 1;
                                    bed_area.width_x = bed_col_low_count - 1;
                                }
                                bed_area.center_x = bed_area.start_x + ((bed_area.width_x + 1) >> 1);
                                bed_area.before_x = bed_area.start_x + ((bed_area.width_x + 2) >> 2);
                                first_col_mark = true;
                                first_col_position = j;
                                bed_col_low_count = 0;
                                zero_count = 0;
                                people_col_status = 7;
                            }
                            else if ((bed_col_low_count < 7) || (zero_count > 0)) {
                                first_col_mark = true;
                                first_col_position = j;
                                bed_col_low_count = 0;
                                zero_count = 0;
                                people_col_status = 8;
                            }


                        }
#ifdef DEBUG_PEOPLE_COL
                        //col_diff
                     //   if (frame_no > 14)
                        printf("$$ The %2d col  %d and diff %d count %d status %d\n", j, col_total[j], col_diff, bed_col_low_count, people_col_status);
#endif
                    } //for (int j = col_first_line


  ///////////////////////////////////////////////////////轉身和移動 //////////////////////////////////////////////////////////////

                    //inside_bed_tmp = (bed_area.start_x != 0) & (bed_area.start_y != 0) & (bed_area.end_x != 0) & (bed_area.end_y != 0);
                    //inside_bed = inside_bed_tmp & (real_bed_mode || ((now_y >= bed_first_line) | (now_y < bed_end_line))) & (people_height < 145);
                    inside_bed_tmp = (no_bed == 0) & (bed_area.start_x != 0) & (bed_area.start_y != 0) & (bed_area.end_x != 0) & (bed_area.end_y != 0);
                    bool now_inside_bed_tmp = ((now_y >= bed_first_line) | (now_y < bed_end_line)) || ((bed_area_pre.area > 250));
                    inside_bed = inside_bed_tmp & (real_bed_mode || now_inside_bed_tmp);// &(people_height < 150);
#ifdef  Wayne_debug
                    printf("\n True is %d", (0 < 1));
                    printf("\n no_bed : %d", no_bed);
                    printf("\n start_x : %d", bed_area.start_x);
                    printf("\n start_y : %d", bed_area.start_y);
                    printf("\n end_x : %d", bed_area.end_x);
                    printf("\n end_y : %d", bed_area.end_y);
                    printf("\n real_bed_mode || now_inside_bed_tmp : %d", (real_bed_mode || now_inside_bed_tmp));
                    printf("\n##############################################");
#endif                   
                    if (inside_bed) is_inside_bed = 1;
                    else is_inside_bed = 0;

                    // cout << "Frame_no " << frame_no << " inside bed " << inside_bed << endl;
                    if ((inside_bed | lie_down_mode) & bed_obj_status) { //20220126
                        //detect bed head
                        bed_area.center_high = grayImage_bed(bed_area.center_y, bed_area.center_x);
                        if (bed_area.center_high > bed_high) bed_area.center_high = bed_high;
                        bed_area.before_high = grayImage_bed(bed_area.before_y, bed_area.before_x);
                        if (bed_area.before_high > bed_high) bed_area.before_high = bed_high;
                        bed_area.area = 0;
                        bed_area.area_low = 0;
                        bed_area.area_middle = 0;
                        bed_area.area_high = 0;
                        bed_area.area_half_high = 0;

                        Mat_<uchar>  now_image = grayImage_now;

                        int half_end = bed_area.start_x + (bed_area.width_x >> 1);
                        int area_high_level = 30;
                        int area_middle_level = 5;

                        if ((bed_high < 129) && max_high_less_240) {
                            area_high_level = 30;
                            area_middle_level = 15;
                        }
                        else if ((bed_high > 128) && max_high_less_240) {
                            area_high_level = 40;
                            area_middle_level = 25;
                        }
                        else if (bed_high > 128) {
                            area_high_level = 30;
                            area_middle_level = 15;
                        }
                        else {
                            area_high_level = 30;
                            area_middle_level = 5;
                        }

                        if (max_high_less_240) {
                            up_high_level = UP_HIGH_LEVEL_240; // 22;
                            down_level_th = DOWN_LEVEL_TH_240;

                        }
                        else {

                            up_high_level = UP_HIGH_LEVEL_250; // 28;
                            down_level_th = DOWN_LEVEL_TH_250;

                        }
                        if (is_engineer) image_last = Mat(60, 80, CV_8UC1, Scalar(0));
                        int bed_middle_start_x = 0;
                        int bed_middle_start_y = 0;
                        int bed_middle_end_x = 0;
                        int bed_middle_end_y = 0;
                        int bed_area_high_value = 0;
                        int bed_area_high_x = 0;
                        int bed_area_high_y = 0;
                        int bed_area_start_x = 0;
                        int bed_area_start_y = 0;
                        int bed_area_end_x = 0;
                        int bed_area_end_y = 0;

                        for (int i = bed_area.start_y; i < (bed_area.end_y + 1); i++) {
                            for (int j = bed_area.start_x; j < bed_area.end_x; j++) {
                                if ((now_image(i, j) < (bed_high + 5)) &   //low 130
                                    (now_image(i, j) > (bed_high - area_middle_level))) { //115
                                    if (is_engineer) image_last.at<uchar>(i, j) = 150;
                                    if (bed_area_start_y == 0) {
                                        bed_area_start_x = j;
                                        bed_area_start_y = i;
                                    }
                                    if (i > bed_area_end_y) {
                                        bed_area_end_x = j;
                                        bed_area_end_y = i;
                                    }
                                    bed_area.area++;
                                    bed_area.area_low++;
                                }
                                else if ((now_image(i, j) <= (bed_high - area_middle_level)) & //middle 116 4
                                    (now_image(i, j) > (bed_high - area_high_level))) {       //90
                                    if (is_engineer) image_last.at<uchar>(i, j) = 200;
                                    if (bed_area_start_y == 0) {
                                        bed_area_start_x = j;
                                        bed_area_start_y = i;
                                    }
                                    if (i > bed_area_end_y) {
                                        bed_area_end_x = j;
                                        bed_area_end_y = i;
                                    }
                                    if (bed_middle_start_y == 0) {
                                        bed_middle_start_x = j;
                                        bed_middle_start_y = i;
                                    }
                                    if (i > bed_middle_start_y) {
                                        bed_middle_end_x = j;
                                        bed_middle_end_y = i;
                                    }
                                    bed_area.area++;
                                    bed_area.area_middle++;

                                }
                                else if (now_image(i, j) <= (bed_high - area_high_level)) { //high                                   
                                    if (bed_area_start_y == 0) {
                                        bed_area_start_x = j;
                                        bed_area_start_y = i;
                                    }
                                    if (i > bed_area_end_y) {
                                        bed_area_end_x = j;
                                        bed_area_end_y = i;
                                    }
                                    if (bed_area_high_value < now_image(i, j))
                                    {
                                        bed_area_high_value = now_image(i, j);
                                        bed_area_high_x = j;
                                        bed_area_high_y = i;
                                    }
                                    if (is_engineer) image_last.at<uchar>(i, j) = 250;
                                    bed_area.area++;
                                    bed_area.area_high++;
                                    if (j < half_end) {
                                        bed_area.area_half_high++;

                                    }
                                }
                            } //for j
//                             if (frame_no == 1300) printf("The end %d\n", i);
                        } //fot i
                        bed_area.area_half = 0;

                        for (int i = bed_area.start2_y; i < bed_area.end2_y; i++) {
                            for (int j = bed_area.start_x; j < bed_area.end_x; j++) {
                                if (now_image(i, j) < (bed_high + 3)) {
                                    bed_area.area_half++;
                                }
                            }
                        }
                        // ### 2022 START #####
                        int OutSideUpAvgPixel = 0;
                        int OutSideUpArea = 0;
                        int OutSideDownAvgPixel = 0;
                        int OutSideDownArea = 0;
                        bool OutSideUpAreaStatus = false;
                        bool OutSideDownAreaStatus = false;
                        // UP AREA  # < 4: means bed is almost on the top location
                        if (bed_first_line > 4)
                        {
                            for (int i = bed_first_line - 3; i < bed_first_line; i++) {
                                for (int j = bed_area.start_x; j < bed_area.end_x; j++) {
                                    OutSideUpAvgPixel = OutSideUpAvgPixel + now_image(i, j);
                                    OutSideUpArea = OutSideUpArea + 1;
                                }
                            }
                            if (OutSideUpArea != 0) OutSideUpAvgPixel = OutSideUpAvgPixel / OutSideUpArea;
                            if (abs(GroundAvgUpPixel - OutSideUpAvgPixel) > 10)  OutSideUpAreaStatus = true;
                        }
                        // DOWN AREA # >58 : means bed is almost on the down location
                        if (bed_end_line < 57)
                        {
                            for (int i = bed_end_line + 1; i < bed_end_line + 3; i++) {
                                for (int j = bed_area.start_x; j < bed_area.end_x; j++) {
                                    OutSideDownAvgPixel = OutSideDownAvgPixel + now_image(i, j);
                                    OutSideDownArea = OutSideDownArea + 1;
                                }
                            }
                            if (OutSideDownArea != 0) OutSideDownAvgPixel = OutSideDownAvgPixel / OutSideDownArea;
                            if (abs(GroundAvgDownPixel - OutSideDownAvgPixel) > 10)  OutSideDownAreaStatus = true;
                        }
                        // On Edge Status Judge 
                        if (bed_first_line == bed_area.start_y || bed_end_line == bed_area.end_y + 1) //2022
                        {
                            EdgeStatus = true;
                            HumanOnEdgeLineCount++;
                        }
                        else  //Not in Edge
                        {
                            EdgeStatus = false;
                            HumanOnEdgeLineCount = 0;
                        }
#ifdef Wayne_debug
                        printf("\nEdgeStatus  : %d", EdgeStatus);
                        printf("\nHumanOnEdgeLineCount  : %d", HumanOnEdgeLineCount);
                        printf("\nOutSideUpAreaStatus  : %d", OutSideUpAreaStatus);
                        printf("\n OutSideDownAreaStatus : %d ", OutSideDownAreaStatus);
                        printf("\n Diff : %d ", abs(GroundAvgDownPixel - OutSideDownAvgPixel));
#endif
                        // ### 2022 END #####
                        bed_area.hight_div_2 = bed_area.start2_y + ((bed_area.width2_y + 1) >> 1);
                        bed_area.width_div_2 = bed_area.start_x + ((bed_area.width_x + 1) >> 1);
                        int tmp_half_max = bed_area.area_half;
                        if (tmp_half_max > bed_area.area_half_max) bed_area.area_half_max = tmp_half_max;
                        int tmp_max = bed_area.area;
                        if (tmp_max > bed_area.area_max) bed_area.area_max = tmp_max;

                        int bed_status = 0;
                        // if (inside_bed) {
                            // int left_begin = bed_first_line + (bed_width >> 1) + 3;
                           //  if (frame_no > 593) printf("The inside bed frame_no is %d\n", frame_no);
                        leave_bed = 0;
                        if (inside_bed_pre == 1) inside_bed_pre = 1;
                        else inside_bed_pre = 0;
                        //  lie_down_high = 15;

                        if (frame_no > 3) {
                            int meddle_end_diff = abs(bed_area_start_x - bed_area_high_x);
                            if (
                                (((bed_area.start_y < (bed_first_line + edge_up_limit)) |
                                    (bed_area.end_y > (bed_end_line - edge_down_limit))))
                                & (bed_area.width_x > 8) & (bed_area.width_y > 5)
                                & (bed_area.area_high > 18) //& (bed_area.area_high < 120)
                                & ((bed_area.area - bed_area_pre.area) < 100) & (bed_area.width_y < 19)
                                & (bed_area.width_x < 27)
                                & (bed_area.area < 360) & (bed_area.area > 119) //119) 126
                              //  & (bed_area.width_x > bed_area.width_y)
                                & (!lie_down_ready) //& !check_no_lr
                                & (meddle_end_diff > 2) & (EdgeStatus == true) //2022
                                & HumanOnEdgeLineCount > 5 //2022
                                & (OutSideUpAreaStatus == true || OutSideDownAreaStatus == true) //2022
                                )
                            {
                                bed_edge = 1;
                                fall_hold = 0;
                                inside_bed_pre = 1;
                                real_bed_mode = 1;
                                is_inside_bed = 0;
                                trun_lie_down = 0;
                                up_lie_down = 0;
                                turn_up = 0;
                                fall_hold = 0;
                                edge_lie_down = true;
                                keep_equal_same_up = 0;
                                if (display_bed_detect) {
                                    sprintf(tmp_status, "on edge");
                                    bed_status = 1;
                                }

                            }
                            else if (((bed_area.width_x <= bed_area_pre.width_x) && (lie_down_ready) &&
                                (abs(bed_area.center_x - bed_area.before_x) < 9)
                                && (bed_area.area <= bed_area_pre.area) && (bed_area.area_high > up_high_level)
                                && (bed_area.area_high > bed_area_pre.area_high) && (!edge_lie_down)
                                && (!((bed_area.width_x >> 1) > bed_area.width_y))
                                ))

                            {
                                if (display_bed_detect) sprintf(tmp_status, "Up-----");
                                bed_edge = 0;
                                edge_lie_down = 0;
                                up_lie_down = true;
                                trun_lie_down = 0;
                                turn_up = 1;
                                //   up_center_x = bed_area.center_x;
                                //   up_center_y = bed_area.center_y;
                                if (bed_area.area_high == bed_area_pre.area_high) {
                                    keep_equal_same_up++;
                                }
                                if (keep_equal_same_up > 5) {
                                    turn_up = 0;
                                    keep_equal_same_up = 0;
                                }

                                lie_down_ready = 0;
                                if (display_bed_detect) bed_status = 3;
                            }
                            else if ((edge_lie_down || up_lie_down) && (!lie_down_ready)
                                // && (bed_area.area_half_high < lie_down_high) && (!lie_down_ready) && (bed_area.area_half_high >0)
                                && (bed_area.area_high < down_level_th)
                                && (bed_area.width_x > (col_width >> 1))
                                && (bed_area.width_y > 12))
                            {
                                if (display_bed_detect) sprintf(tmp_status, "LieDown");
                                EdgeStatus = false; //2022
                                lie_down_ready = 1;
                                up_lie_down = 0;
                                is_lie_down = 1;
                                trun_lie_down = 1;
                                lie_down_mode = 1;
                                real_bed_mode = 1;
                                bed_edge = 0;
                                turn_up = 0;
                                fall_hold = 0;
                                keep_equal_same_up = 0;

                                if (display_bed_detect) bed_status = 4;
                            }
                            else {

                                if (display_bed_detect) {
                                    sprintf(tmp_status, "???????");
                                    bed_status = 7;
                                }
                                if (is_lie_down) {
                                    edge_lie_down = 0;
                                    up_lie_down = 0;
                                    is_lie_down = 0;
                                }
                                trun_lie_down = 0;
                                turn_up = 0;
                                is_inside_bed = 0;
                                if (!other_people_inside & lie_down_mode & (!((now_y >= bed_first_line) & (now_y < bed_end_line))) & (people_height > 130)) {
                                    lie_down_mode = 0;
                                    real_bed_mode = 0;
                                    leave_bed = 1;
                                }
                            }
                            if (display_bed_detect) {
#ifdef LOG_FILE
                                fprintf(fw_log, "****frame_no %4d %s from (%2d,%2d) to (%d, %2d)  width (%2d, %2d)  before(%2d, %2d) = %3d area %3d and center (%d,%d) = %3d status %2d\n",
                                    frame_no, tmp_status, bed_area.start_x, bed_area.start_y, bed_area.end_x, bed_area.end_y, bed_area.width_x,
                                    bed_area.width_y, bed_area.before_x, bed_area.before_y, bed_area.before_high, bed_area.area, bed_area.center_x, bed_area.center_y, bed_area.center_high, bed_status);
                                fprintf(fw_log, "****                     start2_y %2d to end2_y %2d width %2d\n",
                                    bed_area.start2_y, bed_area.end2_y, bed_area.width2_y);
                                fprintf(fw_log, "                         keep left %d  keep right %d turn left %d turb right %d lie_down %d\n", keep_left, keep_right, turn_left, turn_right, trun_lie_down);
                                fprintf(fw_log, "                         same_left %d and same_right %d turn_up %d\n", keep_equal_same_left, keep_equal_same_right, turn_up);
                                fprintf(fw_log, "                         pre fall %d and edge %d inside_bed_pre %d\n", pre_fall, bed_edge, inside_bed_pre);
                                fprintf(fw_log, "                         area %3d and area low %3d area middle %3d area high %3d max area %3d\n",
                                    bed_area.area, bed_area.area_low, bed_area.area_middle, bed_area.area_high, bed_area.area_max);
                                fprintf(fw_log, "                         half area %3d  half high %3d max half area %3d\n",
                                    bed_area.area_half, bed_area.area_half_high, bed_area.area_half_max);
                                fprintf(fw_log, "                         start2_y %d, end2_y %d width2_y %d half max area %d and half area %d\n", bed_area.start2_y, bed_area.end2_y, bed_area.width2_y, bed_area.area_half_max, bed_area.area_half);
                                fprintf(fw_log, "                         hight_div2 = %d, is_lie_down %d edge_lie_down %d up_lie_down %d\n", bed_area.hight_div_2, is_lie_down, edge_lie_down, up_lie_down);
                                fprintf(fw_log, "                         inside %d and noChckLR %d\n", inside_bed, no_check_left_right);
                                fprintf(fw_log, "                         bed_area.start_y %d,  bed_first_line %d, %d  up_high_level %d \n",
                                    bed_area.start_y, bed_first_line, (bed_area.width_x >= (bed_area.width_y << 1)), up_high_level);
                                fprintf(fw_log, "                         pre fall th_right %d th_left %d half_area %d pre_fall_edge %d\n", pre_fall_threshold_right, pre_fall_threshold_left, bed_area.area_half, pre_fall_edge);
                                fprintf(fw_log, "                         high_diff_th %d , max_hight_now %d\n", high_diff_th, max_hight_now);
                                //      fprintf(fw_log,"                         other_people_on_bed %d \n", other_people_on_bed);
                                fprintf(fw_log, "---lie_down_ready:%d---end---inside_bed:%d---\n", lie_down_ready, is_inside_bed);
#else
                                printf("****frame_no %4d %s from (%2d,%2d) to (%d, %2d)  width (%2d, %2d)  before(%2d, %2d) = %3d area %3d and center (%d,%d) = %3d status %2d\n",
                                    frame_no, tmp_status, bed_area.start_x, bed_area.start_y, bed_area.end_x, bed_area.end_y, bed_area.width_x,
                                    bed_area.width_y, bed_area.before_x, bed_area.before_y, bed_area.before_high, bed_area.area, bed_area.center_x, bed_area.center_y, bed_area.center_high, bed_status);
                                printf("****                     start2_y %2d to end2_y %2d width %2d\n",
                                    bed_area.start2_y, bed_area.end2_y, bed_area.width2_y);
                                printf("                         area %3d and area low %3d area middle %3d area high %3d max area %3d\n",
                                    bed_area.area, bed_area.area_low, bed_area.area_middle, bed_area.area_high, bed_area.area_max);
                                printf("                         half area %3d  half high %3d max half area %3d\n",
                                    bed_area.area_half, bed_area.area_half_high, bed_area.area_half_max);
                                printf("                         up %d and edgel %d \n", turn_up, bed_edge);
                                printf("                         start2_y %d, end2_y %d width2_y %d half max area %d and half area %d\n", bed_area.start2_y, bed_area.end2_y, bed_area.width2_y, bed_area.area_half_max, bed_area.area_half);
                                printf("                         lie_down_mode %d trun_lie_down = %d, is_lie_down %d edge_lie_down %d up_lie_down %d\n", lie_down_mode, trun_lie_down, is_lie_down, edge_lie_down, up_lie_down);
                                printf("                         high_diff_th %d , max_hight_now %d\n", high_diff_th, max_hight_now);
                                printf("                         bed_area.start_y %d,  bed_first_line %d,edge_up_limit %d  up_high_level %d \n",
                                    bed_area.start_y, bed_first_line, edge_up_limit, up_high_level);
                                printf("                         bed_area.start_x %d,  bed_end_line %d, edge_down_limit %d \n",
                                    bed_area.start_x, bed_end_line, edge_down_limit);
                                printf("                         sick_house %d , (%d, %d) = %d \n", sick_house, now_x, now_y, people_height);
                                printf("                         inside_bed %d , inside_bed_pre %d inside_bed_tmp %d\n", inside_bed, inside_bed_pre, inside_bed_tmp);
                                printf("                         other_people_inside=%d, lie_down_mode=%d\n", other_people_inside, lie_down_mode);
                                printf("                         middle area diff %d, start(%d, %d) end (%d,%d)\n", meddle_end_diff, bed_middle_start_x, bed_middle_start_y, bed_middle_end_x, bed_middle_end_y);
                                printf("                         area high center (%d, %d)=%d\n", bed_area_high_x, bed_area_high_y, bed_area_high_value);
                                printf("                         area from (%d, %d) to (%d,%d)\n", bed_area_start_x, bed_area_start_y, bed_area_end_x, bed_area_end_y);

                                printf("                         The bed row_y (%d,%d) and col_x (%d, %d) \n", bed_first_line, bed_end_line, col_first_line, col_end_line);
                                printf("                         bed width %d , length %d \n", bed_width, col_width);
                                printf("---lie_down_ready:%d---end---inside_bed:%d- real_bed_mode %d--\n", lie_down_ready, is_inside_bed, real_bed_mode);
#endif
                            }

                        } //frame_no < 3
                    } //inside_bed
                    else {

                        if (bed_edge) {

                            leave_bed = 1;
                            leave_bed_real = 1;
                            inside_bed_pre = 0;
                            lie_down_mode = 0;
                        }
                        else if (leave_bed) {
                            leave_bed = 0;
                            lie_down_mode = 0;
                            leave_bed_real = 1;
                        }
                        bed_edge = 0;

                        if (display_bed_detect == 1) {
#ifdef LOG_FILE
                            fprintf(fw_log, "****frame_no %4d inside %d Leave bed %d amd %d from (%2d,%2d) to (%2d, %2d) people high %d\n",
                                frame_no, inside_bed, inside_bed_pre, leave_bed, bed_area.start_x, bed_area.start_y, bed_area.end_x, bed_area.end_y, people_height);
#else
                            // uchar tmp_key_debug;
                            /*
                             if (leave_bed) {
                               printf("\n");
                                 printf("####frame_no %4d real_bed_mode %d leave_bed_real %d leave_bed %d from (%2d,%2d) to (%2d, %2d) people high (%d, %d)=%d\n",
                                     frame_no, real_bed_mode, leave_bed_real, leave_bed, bed_area.start_x, bed_area.start_y, bed_area.end_x, bed_area.end_y, now_x, now_y, people_height);
                                 printf("#### %d, %d, %d, %d, (%d, %d) \n", inside_bed , inside_bed_tmp , real_bed_mode , now_y,
                                     bed_first_line ,  bed_end_line);
                                 printf("#### area %3d and area low %3d area middle %3d area high %3d max area %3d\n",
                                     bed_area.area, bed_area.area_low, bed_area.area_middle, bed_area.area_high, bed_area.area_max);

                                 printf("\n");
                             }
                             */
#endif
                        }
                    }
                }// !no_bed
                ////
                now_in_bed = ((now_y >= bed_first_line) && (now_y < bed_end_line));
                bool now_side_people = ((now_y > 50) || ((now_x < 10) || (now_x > 70)));
                int bed_first_line_limint = 0;
                int bed_end_line_limint = 59;
                if (bed_first_line > 10)  bed_first_line_limint = bed_first_line - 10;
                else bed_first_line_limint = bed_first_line;
                if (bed_end_line > 50)  bed_end_line_limint = bed_end_line;
                else bed_end_line_limint = bed_end_line + 10;

                if (lie_down_mode & ((people_height > 100) & ((now_y < bed_first_line_limint) || (now_y > bed_end_line_limint)))) {
                    other_people_inside = 1;
                }
                else if (now_side_people || leave_bed) {
                    other_people_inside = 0;
                }
#ifndef net_udp                

                if (show_people_high) {
                    if (inside_bed && (people_height > 145) && other_people_inside && (!now_in_bed)) {
                        sprintf(text, "other %d", people_height);
                    }
                    else sprintf(text, "%d", people_height);
                    putText(resize_img_ir, text, Point(now_4x, now_4y), fontFace, 0.8, cv::Scalar::all(255), thickness, 1);
                    show_people_high = 0;
                }
#endif                
                ////////////////////////////////////////////////////////////////////////////////////////////////////
                if (leave_bed_real && (people_height > 120)) {
                    real_bed_mode = 0;
                    leave_bed_real = 0;
                }
                ////////////////////////////////////////////
                if ((no_bed == 1) || !inside_bed || other_people_inside) {
                    ///跌倒高度差

                    Mat diffImage = Mat::zeros(grayImage_pre.size(), CV_8UC1);
                    Mat_<uchar>  preX_image = grayImage_pre; // grayImage_preX;
                    Mat_<uchar>  nowX_image = grayImage_now_th; //grayImage_nowX;  //now_th
                    Mat_<uchar>  initX_image = grayImage_init;
                    for (int i = 5; i < (grayImage_now.rows - 5); i++) {
                        for (int j = 5; j < (grayImage_now.cols - 5); j++) {
                            if ((nowX_image(i, j) < (initX_image(i, j) - 50)) ||
                                (nowX_image(i, j) > (initX_image(i, j) - 5))
                                ) diffImage.at<uchar>(i, j) = 0;
                            else if ((nowX_image(i, j) > (preX_image(i, j) + 2)) && ((nowX_image(i, j) < (max_hight_now - 10))))
                                diffImage.at<uchar>(i, j) = 250;
                            else diffImage.at<uchar>(i, j) = 0;
                        }
                    }


#ifdef DEBUG_X_FIG

                    namedWindow("diff", WINDOW_AUTOSIZE); //WINDOW_OPENGL);
                    imshow("diff", diffImage);
#endif


                    ////////
                    vector<vector<Point>> contours, contours_now;
                    vector<Vec4i> hierarchy, hierarchy_now;
                    findContours(diffImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


                    int index_tmp = 0;
                    //double
                    int contour_area = 0;
                    box_area = 0;
                    for (int i = 0; i < contours.size(); i++) {
                        contour_area = (int)contourArea(contours.at(i));
                        if (box_area < contour_area) {
                            box_area = contour_area;
                            index_tmp = i;
                        }
                    }
                    bool dist_long = ((abs(now_x - pre_x) > 16) | (abs(now_y - pre_y) > 16));
                    box_area_now = 0;
                    if (box_area > fall_th) {

                        Rect bounding_rect;
                        int center_x, start_x, center_y, start_y;
                        int end_x, end_y;
                        bool startxx = 0, startyy = 0;
                        bool endyy = 0, endxx = 0;

                        bounding_rect = boundingRect(contours[index_tmp]);
                        center_x = bounding_rect.x;
                        start_x = bounding_rect.x;
                        center_y = bounding_rect.y;
                        start_y = bounding_rect.y;

                        if (start_x > edge_level) start_x = start_x - edge_level;
                        else {
                            start_x = 5;
                            startxx = 1;
                        }
                        if (start_y > edge_level) start_y = start_y - edge_level;
                        else {
                            start_y = 5;
                            startyy = 1;
                        }
                        if ((sick_house) && (start_y < 6)) start_y = 5;
                        else if (start_y < 3) start_y = 3;
                        keep_first_stage_x = start_x;
                        keep_first_stage_y = start_y;
                        end_x = center_x + bounding_rect.width + edge_level;
                        end_y = center_y + bounding_rect.height + edge_level;
                        if (end_x >= (diffImage.cols - 2)) {
                            end_x = diffImage.cols - 3;// -5;
                            endxx = 1;
                        }
                        if (end_y >= (diffImage.rows - 2)) {
                            end_y = diffImage.rows - 3;// -5;
                            endyy = 1;
                        }

                        if (real_bed_mode && (end_x > (diffImage.cols - 6))) keep_first_stage_end_x = 72;
                        else if (end_x > (diffImage.cols - 6)) keep_first_stage_end_x = 72;
                        else keep_first_stage_end_x = end_x;

                        if (real_bed_mode && (end_y > (diffImage.rows - 5)) && (end_x > (diffImage.cols - 6))) keep_first_stage_end_y = 52;
                        else if (real_bed_mode && (endyy == 1)) keep_first_stage_end_y = end_y - 2;
                        else if (endyy == 1) keep_first_stage_end_y = end_y - 2;
                        else keep_first_stage_end_y = end_y;

                        bool jump_bed = 0;
                        uchar jump_status = 0;
                        if (no_bed == 0) {
                            if (!((center_y > bed_first_line) || (center_y < (bed_end_line - 1)))) {
                                jump_bed = 1;
                                jump_status = 1;
                            }
                            if ((keep_first_stage_y > bed_first_line) && (keep_first_stage_end_y < bed_end_line)) {
                                jump_bed = 1;
                                jump_status = 2;
                            }
                            else if ((keep_first_stage_y < bed_first_line) && (keep_first_stage_end_y < bed_end_line)) keep_first_stage_end_y = bed_first_line - 1;
                            else if ((keep_first_stage_y > bed_first_line) && (keep_first_stage_end_y > bed_end_line)) keep_first_stage_y = bed_end_line + 2;

                            if (keep_first_stage_end_y <= keep_first_stage_y) {
                                jump_bed = 1;
                                jump_status = 3;
                            }
                            if ((people_height > 150) & (people_height_pre > 150)) {
                                jump_bed = 1;
                                jump_status = 6;
                            }
                        }
                        if ((keep_first_stage_x >= keep_first_stage_end_x) || (keep_first_stage_y >= keep_first_stage_end_y)) {
                            jump_bed = 1;
                            jump_status = 4;
                        }
                        else if ((no_bed == 1) & ((end_x > 77) || (end_x < 2))) {
                            jump_bed = 1;
                            jump_status = 5;
                        }

                        bool check0 = 0;
                        bool check1 = 0;
                        int index_tmp_now = 0;
                        int center_area_x = 0;
                        int center_area_y = 0;

                        Mat now_image_area;
                        now_image_area = Mat::zeros(image.size(), CV_8UC1);
                        if (!jump_bed) {
                            Mat_<uchar>  init_image = grayImage_init;
                            Mat_<uchar>  nowX1_image = grayImage_now_th;
                            int diff_depth = 3;
                            int bed_diff_depth = 0;
                            //    int close_to_ground = 30;
                            other_people2 = (no_bed == 1) &&
                                (((keep_x == 0) & (keep_y == 0)) | ((keep_old_x == 0) & (keep_old_y == 0))) && (other_people_now < 120) &&
                                (other_people_value > 100) && ((abs(now_x - other_people_x) > 10) || (abs(now_y - other_people_y) > 10)) &&
                                (other_people_x >= (keep_first_stage_x - 1)) && (other_people_x <= (keep_first_stage_end_x)) &&
                                (other_people_y >= (keep_first_stage_y - 1)) && (other_people_y <= (keep_first_stage_end_y)) &&
                                (other_people_x > 10) && (other_people_y > 10) && (other_people_x < 69) && (other_people_y < 60)
                                && (abs(other_people_now - other_people_value) < 100);
                            if (max_high_less_240) {
                                if (other_people2) diff_depth = 8; //5
                                else diff_depth = 5;
                                bed_diff_depth = 15;
                            }
                            else {
                                diff_depth = 3;
                                bed_diff_depth = 25;
                            }
                            for (int i = (keep_first_stage_y); i < (keep_first_stage_end_y); i++) {
                                for (int j = (keep_first_stage_x); j < (keep_first_stage_end_x); j++) {
                                    if (!real_bed_mode) {
                                        if ((nowX1_image(i, j) < init_image(i, j)) && (nowX1_image(i, j) < (max_hight_now - 7))) {
                                            int diff_high = init_image(i, j) - nowX1_image(i, j);
                                            if ((diff_high < 30) && (diff_high > diff_depth)) now_image_area.at<uchar>(i, j) = 250;
                                        }
                                        else {
                                            now_image_area.at<uchar>(i, j) = 0; // 250;
                                        }
                                    }//no bed
                                    else {
                                        if (nowX1_image(i, j) < 90) {
                                            jump_bed = 1;
                                            jump_status = 6;
                                        }
                                        else if ((abs(nowX1_image(i, j) - init_image(i, j)) < 5) && (init_image(i, j) > 100))
                                            now_image_area.at<uchar>(i, j) = 0;
                                        else if ((nowX1_image(i, j) > (bed_high - 5)) && (nowX1_image(i, j) < (max_hight_now - bed_diff_depth))) //25
                                            now_image_area.at<uchar>(i, j) = 250;
                                        else    now_image_area.at<uchar>(i, j) = 0;
                                    }//bed
                                }//j
                              //  if (((frame_no > 171) && (frame_no < 176)) || (frame_no > 271)) printf("\n");
                            }//i
                            ///
                            findContours(now_image_area, contours_now, hierarchy_now, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                            box_area_now = 0;
                            index_tmp_now = 0;

                            //double
                            contour_area_now = 0;
                            for (int i = 0; i < contours_now.size(); i++) {
                                contour_area_now = (int)contourArea(contours_now.at(i));
                                if (box_area_now < contour_area_now) {
                                    box_area_now = contour_area_now;
                                    index_tmp_now = i;
                                }
                            }

                            if (box_area_now == 0) {
                                center_area_x = 0;
                                center_area_y = 0;
                            }
                            else {
                                Rect bounding_area_rect;
                                bounding_area_rect = boundingRect(contours_now[index_tmp_now]);
                                center_area_start_x = bounding_area_rect.x;
                                center_area_start_y = bounding_area_rect.y;
                                center_area_x = center_area_start_x + (bounding_area_rect.width >> 1);
                                center_area_y = center_area_start_y + (bounding_area_rect.height >> 1);
                                center_area_end_x = center_area_start_x + (bounding_area_rect.width);
                                center_area_end_y = center_area_start_y + (bounding_area_rect.height);
                            }
                            now_image_area.release();
                            //look for max in fall area
                            max_high_fall_area = 10000;
                            //                           int fall_area_start_x = 0, fall_area_end_x = 0;
                           //                            int fall_area_start_y = 0, fall_area_end_y = 0;
                            if (center_area_start_x > 10) fall_area_start_x = center_area_start_x - 10;
                            else fall_area_start_x = 0;
                            if (center_area_end_x < 70) fall_area_end_x = center_area_end_x + 11;
                            else fall_area_end_x = 80;
                            if (center_area_start_y > 10) fall_area_start_y = center_area_start_y - 10;
                            else fall_area_start_y = 0;
                            if (center_area_end_y < 50) fall_area_end_y = center_area_end_y + 11;
                            else fall_area_end_y = 60;
                            //have bad
                            if (no_bed == 0) {
                                if (fall_area_start_y > bed_end_line) {
                                    if ((bed_end_line + 2) < center_area_start_y) fall_area_start_y = bed_end_line + 2;
                                    else fall_area_start_y = center_area_start_y;
                                }
                                if (fall_area_end_y < bed_first_line) {
                                    if ((bed_first_line + 2) < fall_area_end_y) fall_area_end_y = bed_first_line - 2;
                                    else fall_area_end_y = center_area_end_y;
                                }
                            }
                            for (int i = fall_area_start_y; i < fall_area_end_y; i++) {
                                for (int j = fall_area_start_x; j < fall_area_end_x; j++) {
                                    if ((now_image_1(i, j) < max_high_fall_area) & (now_image_1(i, j) > 10))
                                    {
                                        max_high_fall_area = now_image_1(i, j); // nowX1_image.at<uchar>(i, j);
                                        area_high_x = j;
                                        area_high_y = i;
                                    }
                                    //   if (frame_no == 4628) printf("%3d ", now_image_1(i, j));
                                }
                                //  if (frame_no == 4628) cout << "\n";
                            }
                            /*
                            printf("Debug info The fall area (%d,%d) ,(%d,%d) ,(%d,%d) =%d\n" ,
                            fall_area_start_x,fall_area_start_y, fall_area_end_x, fall_area_end_y,
                            area_high_x, area_high_y, max_high_fall_area);
                            */
                            check0 = (max_high_fall_area > 100) || ((center_area_end_x - center_area_start_x) > 29);// &(area_high_x < 70);

                            ///////////
                            int first_stage_y_th = 5;
                            if (no_bed == 0) first_stage_y_th = 7;
                            if ((box_area > 35) & (box_area_now > 20)) { //12
                                if (!real_bed_mode) {
                                    if (((endxx && (keep_first_stage_end_x > 73)) || (endyy && (keep_first_stage_end_y > 53)) ||
                                        (startxx && (keep_first_stage_x < 5)) || (startyy && (keep_first_stage_y < first_stage_y_th))) && (box_area_now < 110)) {
                                        check1 = 0;
                                    }
                                    else check1 = 1;
                                }
                                else {
                                    if (box_area_now > 50)    check1 = 1;
                                    else check1 = 0;
                                }
                            }
                            else if ((box_area < 36) && (box_area > 26)) {
                                if (box_area_now > (box_area * 2)) check1 = 1;
                                else check1 = 0;
                            }
                            else if ((box_area < 27) && (box_area > 19)) {
                                if ((keep_first_stage_x < 6) || (keep_first_stage_y < first_stage_y_th)) {
                                    if ((no_bed == 1) && (box_area_now >= (box_area * 7))) check1 = 1;
                                    else if ((no_bed == 0) && (box_area_now >= (box_area * 5))) check1 = 1;
                                    else check1 = 0;
                                }
                                else if (box_area_now >= (box_area * 3)) check1 = 1;
                            }
                            else if ((box_area < 20) && (box_area > 11)) {
                                if ((!max_high_less_240) && (box_area_now > 150) && (people_height >= people_height_pre)) check1 = 0;
                                else if ((keep_first_stage_x < 6) || (keep_first_stage_y < first_stage_y_th)) {
                                    if (real_bed_mode && (box_area_now >= (box_area * 2))) check1 = 1;
                                    else if (box_area_now >= (box_area * 7)) check1 = 1;
                                    else check1 = 0;
                                }
                                else if ((box_area_now * 10) >= (box_area * 38)) {
                                    check1 = 1;
                                }
                                else check1 = 0;
                            }
                            else if (box_area < 12) {
                                if ((!max_high_less_240) && (box_area_now > 200)) check1 = 0;
                                else if ((!max_high_less_240) && (box_area_now > 150) && (people_height >= people_height_pre)) check1 = 0;
                                else if ((center_x > 20) && (center_x < 58) && (center_y > 15) && (center_y < 45)
                                    && (box_area_now > 40) && (box_area > 7)) check1 = 1;
                                else if (real_bed_mode && (box_area_now > 40) && (center_y < 51) && (center_y > 10) &&
                                    (keep_first_stage_end_x < 72)) check1 = 1;
                                else if ((box_area_now >= (box_area * 11)) && (box_area_now <= (box_area * 25))) check1 = 1;
                                else if (dist_long & (box_area_now >= (box_area * 4))) check1 = 1;
                                else check1 = 0;

                            }
                            else check1 = 0;
                            // } 
                             // if (box_area_now == 0)
                        } //jump
                    //if (box_area_now != 0) {
                        int people_high_diff = 0;
                        if (people_height < people_height_pre) people_high_diff = people_height_pre - people_height;
                        int people_high_diff_1 = 0;
                        if (people_height_pre < people_height_pre2) people_high_diff_1 = people_height_pre2 - people_height_pre;
                        int people_high_diff_2 = 0;
                        if (people_height < people_height_pre2) people_high_diff_2 = people_height_pre2 - people_height;
                        int move_out_edge = 0;

                        if ((now_x > 70) || (now_x < 10) || (now_y < 11) || (now_y > 65))
                            move_out_edge = (((people_high_diff < 20) & (people_high_diff_1 < 20) & (people_high_diff_2 < 30)));
                        bool check23 = 0;
                        if ((abs(now_x - pre_x) < 2) && (abs(now_y - pre_y) < 2) && ((abs(center_x - now_x) > 10) || (abs(center_y - now_y) > 10))
                            && (box_area > 18) && (box_area_now > 100) && (box_area_pre > 18) && (box_area_now_pre > 100) && (people_height < 100)) check23 = 1;
                        bool other_people = ((keep_old_x == 0) && (keep_old_y == 0) && (abs(now_x - pre_x) < 10) && (abs(now_y - pre_y) < 10) && (people_height > 140));
                        bool special_issue = (((people_height < 126) && (now_x < 70)) && (!((people_height < 65) && (now_x > 68))));
                        bool check2 = (check23 | real_bed_mode) | ((special_issue && !no_people) || ((box_area_now > 200) &&
                            (box_area > 60) && other_people) || other_people2) & (!move_out_edge);
                        //
                        singal_person = (keep_old_x >= (keep_first_stage_x - 1)) && (keep_old_x <= (keep_first_stage_end_x + 1)) &&
                            (keep_old_y >= (keep_first_stage_y - 1)) && (keep_old_y <= (keep_first_stage_end_y + 1)) && (people_height < 100);// 110 &(people_high_diff < 40);
                        //20201110 change for bed
                        bool signal_person1 = dist_long & (people_height_pre >= people_height) & (people_height < 130);
                        //(people_height < 100);// 110 &(people_high_diff < 40);
                    //20201110 change for bed 
                        bool check3_bed = real_bed_mode && !((now_y >= bed_first_line) && (now_y <= bed_end_line)) && (people_height < 110);
                        bool check3 = (check23 | check3_bed) | ((box_area > 70) & (box_area_now > 180)) |
                            ((!real_bed_mode) & (singal_person | other_people2 | signal_person1));
                        bool too_max_area = (box_area > 37) && (box_area_now > 152);
                        bool check4 = (real_bed_mode) || (((keep_old_value > 0) && (keep_old_x != 0) && (keep_old_y != 0)) &&
                            (!((keep_old_value < 120) && (keep_old_x > 68) && (keep_old_y < 11))) || other_people ||
                            too_max_area);
                        int center_x_th = 5;
                        if ((pre_x < 11) && (now_x < 13) && (pre_y > 48) && (people_height == people_height_pre)) center_x_th = 6;
                        else if (other_people2) center_x_th = 10;
                        bool check5 = ((real_bed_mode) & ((center_y == center_area_y) | (!((center_area_y > bed_first_line) && (center_area_y < bed_end_line)))))
                            | ((!real_bed_mode) & ((center_x > center_x_th) & (center_x < 75) & (center_y > 4) & (center_y < 51)));
                        //(center_x > 3) (center_y > 8)
                        int old_now_value = grayImage_now.at<uchar>(keep_old_y, keep_old_x);
                        int old_pre_value = grayImage_pre.at<uchar>(keep_old_y, keep_old_x);
                        int old_pre_pre_value = grayImage_pre_pre.at<uchar>(keep_old_y, keep_old_x);

                        bool check6 = ((no_bed == 0) & ((!inside_bed) | other_people_inside)) | ((no_bed == 1) & ((abs(old_now_value - old_pre_value) > 4) | (box_area_now > 124) | other_people2));// &!inside_bed;
                        bool check7 = !((keep_old_x == pre_x) && (keep_old_y == pre_y) && (keep_old_x != 0) && (keep_old_y != 0));
                        if (check0 && check1 && check2 && check3 && check4 && check5 && check6 & check7) {

                            bounding_rect_now = boundingRect(contours_now[index_tmp_now]);
#ifndef net_udp                            
                            XX_rise_x = (bounding_rect_now.x) * 4 - 10;
                            XX_rise_y = (bounding_rect_now.y) * 4 - 10;
#endif                            
                            if (show_fall_message) {
#ifdef LOG_FILE
                                fprintf(fw_log, "------The frame no is %d and check1:%d check2:%d check3:%d check4:%d-check5 %d-check6 %d ------\n",
                                    frame_no, check1, check2, check3, check4, check5, check6); // , check7);
                                fprintf(fw_log, "The diff value is %d and area %d and check high %d\n", box_area, box_area_now, check_point_high);
                                fprintf(fw_log, "The high signal first_change %d, now(%d, %d), pre(%d,%d) and people_height %d pre %dnow max %d\n",
                                    first_change, now_x, now_y, pre_x, pre_y, people_height, people_height_pre, max_hight_now);
                                fprintf(fw_log, "The value now is %d and pre is %d and old is %d\n", tmpNowMin, tmpMin, keep_old_value);
                                fprintf(fw_log, "The high now: %d and th: %d move %d\n", max_hight_now, high_diff_th, move_out_edge);
                                fprintf(fw_log, "The value old1 is %d\n", old_now_value);
                                fprintf(fw_log, "The value pre1 is %d\n", old_pre_value);

                                fprintf(fw_log, "The people high diff %d %d  2 %d\n", people_high_diff, people_high_diff_1, people_high_diff_2);
                                fprintf(fw_log, "The other people %d %d singal_person %d\n", other_people, other_people2, singal_person);
                                fprintf(fw_log, "The other people (%d, %d)\n", other_people_x, other_people_y);
                                fprintf(fw_log, "The old is       (%d, %d)\n", keep_old_x, keep_old_y);
                                fprintf(fw_log, "The firstage str (%d, %d)\n", keep_first_stage_x, keep_first_stage_y);
                                fprintf(fw_log, "The firstage end (%d, %d)\n", keep_first_stage_end_x, keep_first_stage_end_y);
                                fprintf(fw_log, "The center (%d, %d)\n", center_x, center_y);
                                fprintf(fw_log, "The center area center(%d, %d)\n", center_area_x, center_area_y);
                                fprintf(fw_log, "The center area start(%d, %d) end (%d,%d)\n", center_area_start_x, center_area_start_y, center_area_end_x, center_area_end_y);
                                fprintf(fw_log, "The end area (%d, %d) jump %d %d max depth %d\n", end_x, end_y, jump_bed, jump_status, max_hight_now);
                                fprintf(fw_log, "The bed mode %d pre_fall_leave %d check23 %d\n", real_bed_mode, pre_fall_leave, check23);
                                fprintf(fw_log, "====== %d ======%d====%d===%d===\n", people_here_count, real_bed_mode, move_out_edge, bed_height);
#else
                                printf("------The frame no is %d and check1:%d check2:%d check3:%d check4:%d-check5 %d-check6 %d move_out_edge %d------\n",
                                    frame_no, check1, check2, check3, check4, check5, check6, move_out_edge); // , check7);
                                printf("The fall area high (%d,%d) = %d\n", area_high_x, area_high_y, max_high_fall_area);
                                printf("The diff value is %d and area %d and check high %d\n", box_area, box_area_now, check_point_high);
                                printf("The high signal first_change %d, now(%d, %d), pre(%d,%d) and people_height %d pre %d\n",
                                    first_change, now_x, now_y, pre_x, pre_y, people_height, people_height_pre);
                                printf("The max depth %d min depth is %d and pre is %d and old is %d\n", max_hight_now, tmpNowMin, tmpMin, keep_old_value);
                                printf("The move %d center_x_th %d high th %d sick_house %d\n", move_out_edge, center_x_th, high_diff_th, sick_house);
                                printf("The people high diff %d %d  P2 %d\n", people_high_diff, people_high_diff_1, people_high_diff_2);
                                printf("The other people %d %d singal_person %d signal_person1 %d\n", other_people, other_people2, singal_person, signal_person1);
                                // other_people_now, other_people_pre
                                printf("The other people (%d, %d) = %d now %d\n", other_people_x, other_people_y, other_people_value, other_people_now);
                                printf("The old is       (%d, %d) = %d \n", keep_old_x, keep_old_y, old_now_value); // , old_people_height);
                                printf("The old pre is   (%d, %d) = %d \n", keep_old_x, keep_old_y, old_pre_value); // , old_people_height_pre);
                                printf("The old pre pre  (%d, %d) = %d \n", keep_old_x, keep_old_y, old_pre_pre_value); // , old_people_height_pre_pre);
                                printf("The firstage str (%d, %d)\n", keep_first_stage_x, keep_first_stage_y);
                                printf("The firstage end (%d, %d)\n", keep_first_stage_end_x, keep_first_stage_end_y);
                                printf("The center (%d, %d)\n", center_x, center_y);
                                printf("The center area center(%d, %d)\n", center_area_x, center_area_y);
                                printf("The center area start(%d, %d) end (%d,%d)\n", center_area_start_x, center_area_start_y, center_area_end_x, center_area_end_y);
                                printf("The end area (%d, %d) jump %d %d max depth %d\n", end_x, end_y, jump_bed, jump_status, max_hight_now);
                                printf("The real_bed_mode %d  check23 %d check3_bed %d \n", real_bed_mode, check23, check3_bed);
                                printf("***  other_people_inside %d lie_down_mode %d\n", other_people_inside, lie_down_mode);
                                printf("***  bed_first (%d,%d) bed_end (%d,%d)\n", col_first_line, bed_first_line, col_end_line, bed_end_line);
                                printf("====== %d ======%d====%d===%d===\n", people_here_count, real_bed_mode, move_out_edge, inside_bed);
#endif
                            }
                            fall_true = 1;
                            fall_hold = 1;
                            keep_old_x = 0;// now_x;
                            keep_old_y = 0;// now_y;
                            keep_old_value = 0;// tmpNowMin;
                            people_here_count = 0;
                            //
                            trun_lie_down = 0;
                            if ((other_people_inside == 1) & ((center_area_start_y < now_y) || (center_area_end_y > now_y))) {
                                lie_down_mode = 1;
                            }
                            else {
                                lie_down_mode = 0;
                            }
                            lie_down_ready = 0;
                            turn_up = 0;
                            bed_area.start_x = 0;
                            bed_area.end_x = 0;
                            bed_area.start_y = 0;
                            bed_area.end_y = 0;
                        }  //if (box_area > fall_th) 
                        else {
                            if (show_fall_message_pre) {
#ifdef LOG_FILE
                                fprintf(fw_log, "--->The frame no is %d and check1:%d check2:%d check3:%d check4:%d-check5 %d check6 %d -------\n", frame_no, check1, check2, check3, check4, check5, check6); // , check7);
                                fprintf(fw_log, "---> The right diff value is %d and area  %d and check high %d\n", box_area, box_area_now, check_point_high);
                                fprintf(fw_log, "--->The high signal first_change %d, now(%d, %d), pre(%d,%d) people_height %d pre %d\n",
                                    first_change, now_x, now_y, pre_x, pre_y, people_height, people_height_pre);
                                fprintf(fw_log, "--->The value now is %d and pre is %d and old is %d\n", tmpNowMin, tmpMin, keep_old_value);
                                fprintf(fw_log, "--->The value old1 now %d pre %d\n", old_now_value, old_pre_value);
                                fprintf(fw_log, "--->The other people (%d, %d)\n", other_people_x, other_people_y);
                                fprintf(fw_log, "--->The old (%d, %d) is %d check point %d\n", keep_old_x, keep_old_y, keep_old_value, check_point_high);
                                fprintf(fw_log, "--->The people high diff %d %d singal %d\n", people_high_diff, people_high_diff_1, people_high_diff_2);
                                fprintf(fw_log, "--->The other people %d %d singal %d\n", other_people, other_people2, singal_person);
                                fprintf(fw_log, "--->The high now: %d and th: %d move_out %d\n", max_hight_now, high_diff_th, move_out_edge);
                                fprintf(fw_log, "--->The firstage str (%d, %d)\n", keep_first_stage_x, keep_first_stage_y);
                                fprintf(fw_log, "--->The firstage end (%d, %d)\n", keep_first_stage_end_x, keep_first_stage_end_y);
                                fprintf(fw_log, "--->The center (%d, %d)\n", center_x, center_y);
                                fprintf(fw_log, "--->The end area (%d, %d)\n", end_x, end_y);
                                fprintf(fw_log, "--->The center area center (%d, %d)\n", center_area_x, center_area_y);
                                fprintf(fw_log, "--->The center area start(%d, %d) end (%d,%d)\n", center_area_start_x, center_area_start_y, center_area_end_x, center_area_end_y);
                                fprintf(fw_log, "--->The end area (%d, %d) jump %d %d\n", end_x, end_y, jump_bed, jump_status);
                                fprintf(fw_log, "--->check23 %d, real_bed_mode:%d pre_fall_leave %d\n", check23, real_bed_mode, pre_fall_leave);
                                fprintf(fw_log, "--->people_here_count %d and no_people %d %d\n", people_here_count, no_people, inside_bed);
                                fprintf(fw_log, "*******end  %d****%d****%d*\n", bed_height, move_out_edge, real_bed_mode);
#else

                                printf("--->The frame no is %d and check0 %d check1:%d check2:%d check3:%d check4:%d-check5 %d check6 %d -------\n",
                                    frame_no, check0, check1, check2, check3, check4, check5, check6); // , check7);
                                printf("--->The fall area high (%d,%d) = %d\n", area_high_x, area_high_y, max_high_fall_area);
                                printf("---> The right diff value is %d and area  %d and check high %d\n", box_area, box_area_now, check_point_high);
                                printf("--->The high signal first_change %d, now(%d, %d), pre(%d,%d) people_height %d pre %d\n",
                                    first_change, now_x, now_y, pre_x, pre_y, people_height, people_height_pre);
                                printf("--->The value now is %d and pre is %d and old is %d\n", tmpNowMin, tmpMin, keep_old_value);
                                printf("--->The value old1 now %d pre %d\n", old_now_value, old_pre_value);
                                printf("--->The other people (%d, %d)\n", other_people_x, other_people_y);
                                printf("--->The old (%d, %d) is %d check point %d\n", keep_old_x, keep_old_y, keep_old_value, check_point_high);
                                printf("--->The people high diff %d %d diff_2 %d\n", people_high_diff, people_high_diff_1, people_high_diff_2);
                                printf("--->The other people %d %d singal %d signal_person1 %d\n", other_people, other_people2, singal_person, signal_person1);
                                printf("--->The other people (%d, %d) = %d now %d\n", other_people_x, other_people_y, other_people_value, other_people_now);
                                printf("--->The high now: %d and th: %d move_out %d\n", max_hight_now, high_diff_th, move_out_edge);
                                printf("--->The firstage str (%d, %d)\n", keep_first_stage_x, keep_first_stage_y);
                                printf("--->The firstage end (%d, %d)\n", keep_first_stage_end_x, keep_first_stage_end_y);
                                printf("--->The center (%d, %d)\n", center_x, center_y);
                                printf("--->The end area (%d, %d)\n", end_x, end_y);
                                printf("--->The center area center (%d, %d)\n", center_area_x, center_area_y);
                                printf("--->The center area start(%d, %d) end (%d,%d)\n", center_area_start_x, center_area_start_y, center_area_end_x, center_area_end_y);
                                printf("--->The end area (%d, %d) jump %d %d\n", end_x, end_y, jump_bed, jump_status);
                                printf("--->check23 %d, real_bed_mode:%d \n", check23, real_bed_mode);
                                printf("--->people_here_count %d and no_people %d %d\n", people_here_count, no_people, inside_bed);
                                printf("---> (check23 %d | check3_bed %d) | ((!real_bed_mode %d) & (singal_person %d | other_people2 %d))\n",
                                    check23, check3_bed, real_bed_mode, singal_person, other_people2);
                                printf("--->  other_people_inside %d lie_down_mode %d\n", other_people_inside, lie_down_mode);
                                printf("*******end  %d****%d****%d*****dist%d\n", bed_height, move_out_edge, real_bed_mode, dist_long);

#endif
                            }
                        } //fall 1 ok and fall2 not

                    } //(box_area > fall_th)
#ifdef DEBUG_FALL_SLOW     
                    if (frame_no > 2300) {
                        printf(">>>>>>>> The %d right diff value is %d and %d high %d\n", frame_no, box_area, fall_th, people_height);
                        printf(">>>>>>>> The high signal first_change %d, now(%d, %d)= %d, pre(%d,%d)=%d, old (%d, %d)=%d\n",
                            first_change, now_x, now_y, tmpNowMin, pre_x, pre_y, tmpMin, keep_old_x, keep_old_y, keep_old_value);
                        printf(">>>>>>> people_here_count %d and no_people %d inside_bed %d\n", people_here_count, no_people, inside_bed);
                        printf(">>>>>>> max_hight_now %d and no_people %d inside_bed %d\n", max_hight_now, no_people, inside_bed);
                    }
                    // printf(">>>>>>>> The end area jump %d %d\n", jump_bed, jump_status);
#endif
                } // (no_bed | !inside_bed)

                else {   //if (box_area > fall_th)
                    keep_old_x = 0;
                    keep_old_y = 0;
                }  //if (box_area > fall_th)
                box_area_pre = box_area;
                box_area_now_pre = box_area_now;
                ////////////////////////////////////////////////////////////////////////////
                                //detect people
                if (fall_hold || real_bed_mode) people_counnt_enable = 0;
                else if ((people_height < 100) & (frame_no > 5)) {
                    people_counnt_enable = 1;
#ifdef PEOPLE_HAVE
                    printf("The people_counnt_enable %d\n", frame_no);
#endif
                }
                else people_counnt_enable = 0;

                //if (!people_counnt_enable) people_here_count = 0;
                if (people_counnt_enable && (people_here_count == people_frame) && (people_height < 90)) people_here_count = people_frame;
                else if (people_counnt_enable && (no_bed == 1) && (people_height < 90)) people_here_count = people_here_count + 1;
                else if (people_counnt_enable && (no_bed == 0) && (!real_bed_mode) && ((people_height < 110) || (now_in_bed))) people_here_count = people_here_count + 1;
                else people_here_count = 0;

                if (draw_frame == 0)     people_frame = 140;
                else if (draw_frame < 2) people_frame = 140;
                else if (draw_frame < 5) people_frame = 65;
                else people_frame = 60;

                if (people_counnt_enable)
                    no_people = (people_here_count == (people_frame)) && (!fall_true);
                else no_people = 0;

                /* if (no_people)          no_people_act = 1;
                else*/ if (have_people) {
                //   no_people_act = 0;
                    fall_hold = 0;
                }
                if (have_people)    have_people_act = 1;
                else if (no_people)     have_people_act = 0;

                have_people = (!no_people) && (people_height > 120) && (!real_bed_mode) & (frame_no > 5); //& !have_people_act
                 /////////

                //////

                bool have_people_true = (have_people && (!have_people1));
                bool no_people_show = no_people && (!no_people_pre);
                ////////////////
#ifdef PEOPLE_HAVE
                //if (!people_counnt_enable) {
                if (frame_no > 4560) {
                    printf("================================================================================\n");
                    printf("The frame_no %d, no_people %d, people_here_count %d people_counnt_enable %d\n",
                        frame_no, no_people, people_here_count, people_counnt_enable);
                    printf("The people_high (%d, %d) = %d tmpNowMin %d\n", now_x, now_y, people_height, tmpNowMin);
                    printf("The people %d, have_people1 %d, \n", have_people, have_people1);
                }
#endif
                bool turn_up_tmp = turn_up & !turn_up_pre;
                bool other_people_inside_tmp = (!other_people_inside_pre) & other_people_inside;
                /*  if (init_data == 1) sprintf(text_show, "init data ");
                  else
                  */
                sprintf(text_show, "          ");
                if (fall_true || trun_lie_down || turn_up_tmp || bed_edge || leave_bed ||
                    no_people_show || have_people_true || other_people_inside_tmp)
                {
#ifdef Compare_LOG
                    fprintf(ComPfw_log, "Frame_no:%4d, ", frame_no);
                    if (fall_true)                    fprintf(ComPfw_log, "fall_true\n");
                    else if (!fall_hold) {
                        if (pre_fall)         fprintf(ComPfw_log, "pre_fall\n");
                        else if (trun_lie_down)    fprintf(ComPfw_log, "trun_lie_down\n");
                        else if (turn_right)       fprintf(ComPfw_log, "turn_right\n");
                        else if (turn_left)        fprintf(ComPfw_log, "turn_left\n");
                        else if (turn_up)          fprintf(ComPfw_log, "turn_up\n");
                        else if (bed_edge)         fprintf(ComPfw_log, "bed_edge\n");
                        else if (leave_bed)        fprintf(ComPfw_log, "leave_bed\n");
                        else if (no_people_show)        fprintf(ComPfw_log, "no_people\n");
                        else if (have_people_true) fprintf(ComPfw_log, "have_people_true\n");
                        else if (bed_head_up & bed_head_up_pre) fprintf(ComPfw_log, "Bed Head Up\n");
                        else if (bed_head_up & !bed_head_up_pre) fprintf(ComPfw_log, "Bed Head Down\n");
                    }
#endif
#ifdef net_udp
                    Rect temp_r(bounding_rect_now.x, bounding_rect_now.y, bounding_rect_now.width, bounding_rect_now.height);
#else
                    cvtColor(resize_img, C_grayImage_XX_rise, CV_GRAY2RGB);
                    int XX_rise_yyy = bounding_rect_now.y * 4;
                    int XX_rise_width = bounding_rect_now.width * 4;
                    int XX_rise_height = bounding_rect_now.height * 4;
                    Rect temp_r(XX_rise_x, XX_rise_yyy, XX_rise_width, XX_rise_height);

                    if (frame_no_is == 1) {
                        if (fall_true) {
                            //  if (contour_area_now != 0)
                            rectangle(C_grayImage_XX_rise, temp_r, Scalar(250, 50, 50), 2, 1, 0);
                            if (other_people2 && !singal_person) sprintf(text_show, "Falling %d pp", frame_no);
                            else                                 sprintf(text_show, "Falling %d", frame_no);
                        }
                        else if ((!fall_hold) && trun_lie_down)                     sprintf(text_show, "lie down %d", frame_no);
                        else if ((!fall_hold) && turn_up)                           sprintf(text_show, "Up %d", frame_no);
                        else if ((!fall_hold) && bed_edge)                          sprintf(text_show, "Edge %d", frame_no);
                        else if ((!fall_hold) && leave_bed)                         sprintf(text_show, "Leave %d", frame_no);
                        else if ((!fall_hold) && no_people_show)                    sprintf(text_show, "no people %d", frame_no);
                        else if ((!fall_hold) && have_people_true && (!inside_bed || (inside_bed && !bed_obj_status))) sprintf(text_show, "people %d", frame_no); //20220126 means (not in bed) or (in bed but have a small pixel change)
                        else if ((!fall_hold) && other_people_inside_tmp)           sprintf(text_show, "other people %d", frame_no);
                    }
                    else {

                        if (fall_true) {
#ifdef net_udp 
                            rectangle(image_send, temp_r, Scalar(0, 0, 244), 0.8, 1, 0);
#else                            
                            rectangle(C_grayImage_XX_rise, temp_r, Scalar(15, 251, 244), 2, 1, 0);
#endif                        
                            sprintf(text_show, "Falling");
                        }
                        else if ((!fall_hold) && trun_lie_down)                             sprintf(text_show, "lie down ");
                        //else if ((!fall_hold) && inside_bed)                                sprintf(text_show, "On Bed ");
                        else if ((!fall_hold) && turn_up)                                   sprintf(text_show, "Up ");
                        else if ((!fall_hold) && bed_edge)                                  sprintf(text_show, "Edge ");
                        else if ((!fall_hold) && leave_bed)                                 sprintf(text_show, "Leave ");
                        //else if ((!fall_hold) && (no_bed==0) && (frame_no < 5))             sprintf(text_show, "bed ");
                        else if ((!fall_hold) && no_people_show)                            sprintf(text_show, "no people");
                        else if ((!fall_hold) && have_people_true && (!inside_bed || (inside_bed && !bed_obj_status)))           sprintf(text_show, "people"); //20220126
                        else if ((!fall_hold) && other_people_inside_tmp)                   sprintf(text_show, "other people");
                    }
                    if (fall_true) {
                        if (XX_rise_yyy < 30) XX_rise_y = (XX_rise_yyy + XX_rise_height + 25);
                        if (XX_rise_x > 120) {
                            if (frame_no_is)  XX_rise_x = XX_rise_x - 110;
                            else XX_rise_x = XX_rise_x - 40;
                        }
                        else XX_rise_x = 20;
                        putText(C_grayImage_XX_rise, text_show, Point(XX_rise_x, XX_rise_y), FONT_HERSHEY_TRIPLEX, 1.0, Scalar(18, 12, 255), 1, 8);
                    }
                    else putText(C_grayImage_XX_rise, text_show, Point(100, 40), fontFace, 0.8, cv::Scalar::all(255), thickness, 1);
#endif                    
                } //if (pre_fall | fall_true
#ifndef net_udp
                String_tmp = IMAGE_PATH;
                bool people_have_1 = first_change && (people_height > 140) && no_bed;
                if (fall_true)                                              grayImage_GIS = imread(String_tmp + "gis_320x240_fall_red.jpg", IMREAD_COLOR);
                else if (trun_lie_down)                                      grayImage_GIS = imread(String_tmp + "gis_320x240_lie.jpg", IMREAD_COLOR);
                else if (turn_up)                                            grayImage_GIS = imread(String_tmp + "gis_320x240_up.jpg", IMREAD_COLOR);
                else if (bed_edge)                                           grayImage_GIS = imread(String_tmp + "gis_320x240_edge.jpg", IMREAD_COLOR);
                else if (leave_bed)                                          grayImage_GIS = imread(String_tmp + "gis_320x240_leave_bed.jpg", IMREAD_COLOR);
                else if ((no_bed == 1) && no_people_show)                      grayImage_GIS = imread(String_tmp + "gis_320x240_no_people.jpg", IMREAD_COLOR);
                else if ((no_bed == 0) && no_people_show)                      grayImage_GIS = imread(String_tmp + "gis_320x240_bed_no_people.jpg", IMREAD_COLOR);
                else if ((no_bed == 1) && have_people_true)                    grayImage_GIS = imread(String_tmp + "gis_320x240_people.jpg", IMREAD_COLOR);
                else if ((no_bed == 0) && have_people_true && (!inside_bed || (inside_bed && !bed_obj_status)))     grayImage_GIS = imread(String_tmp + "gis_320x240_bed_people.jpg", IMREAD_COLOR); //20220126
                else if (people_have_1)                                      grayImage_GIS = imread(String_tmp + "gis_320x240_people.jpg", IMREAD_COLOR);
                else  if (frame_no < 4) {
                    if (no_bed == 1)                                           grayImage_GIS = imread(String_tmp + "gis_320x240.jpg", IMREAD_COLOR);
                    else                                                     grayImage_GIS = imread(String_tmp + "gis_320x240_bed.jpg", IMREAD_COLOR);
                }
                else if ((no_bed == 0) && other_people_inside_tmp)           grayImage_GIS = imread(String_tmp + "gis_320x240_bed_people.jpg", IMREAD_COLOR);
#endif
                ////////////////////////////////////////////
#ifdef HTTP_ENABLE
                int send_touchlife_no = 4;
                if (fall_true)       send_touchlife_no = 0;
                else if (no_bed & no_people)  send_touchlife_no = 1;
                else if (people_have_1) send_touchlife_no = 2;
                else send_touchlife_no = 4;
                if ((send_touchlife_no < 4) & (send_touchlife_no != send_touchlife_no_pre)) {
                    time(&now);
                    localtime_s(&t, &now);
                    //char* year, * date, time*;
                    year_int = (t.tm_year + 1900);
                    month_int = (t.tm_mon + 1);
                    day_int = (t.tm_mday);
                    hour_int = t.tm_hour;
                    min_int = t.tm_min;
                    year_s = to_string(year_int);
                    month_s = to_string(month_int);
                    if (month_int < 10)  month_s = "0" + month_s;
                    day_s = to_string(day_int);
                    if (day_int < 10) day_s = "0" + day_s;

                    hour_int = t.tm_hour;
                    min_int = t.tm_min;
                    sec_int = t.tm_sec;
                    hour_s = to_string(hour_int);
                    if (hour_int < 10) hour_s = "0" + hour_s;
                    min_s = to_string(min_int);
                    if (min_int < 10) min_s = "0" + min_s;
                    sec_s = to_string(sec_int);
                    if (sec_int < 10) sec_s = "0" + sec_s;
                    char date_name[80];
                    char time_name[80];
                    char image_name[80];
                    strcpy(date_name, &year_s[0]);
                    strcat(date_name, "-");
                    strcat(date_name, &month_s[0]);
                    strcat(date_name, "-");
                    strcat(date_name, &day_s[0]);

                    strcpy(time_name, &hour_s[0]);
                    strcat(time_name, ":");
                    strcat(time_name, &min_s[0]);
                    strcat(time_name, ":");
                    strcat(time_name, &sec_s[0]);

                    strcpy(image_name, "smart10005_");
                    strcat(image_name, &year_s[0]);
                    strcat(image_name, &month_s[0]);
                    strcat(image_name, &day_s[0]);
                    strcat(image_name, &hour_s[0]);
                    strcat(image_name, &min_s[0]);
                    strcat(image_name, &sec_s[0]);
                    //                    const char* msg_send = "createEvent?timestamp=%sT%s&caption=%s&state=Active&submit=submit";
                                       // const char* msg_send = "timestamp=%sT%s&caption=%s&state=Active";
                    const char* msg_send = "http://admin:sentry256@1.34.64.193:7001/api/createEvent?timestamp=%sT%s&caption=%s&state=Active";
                    char msg[256] = { 0 };
                    sprintf(msg, msg_send, date_name, time_name, text_show);
                    printf("Http send command is %s\n", msg);
                    curl = curl_easy_init();
                    if (curl) {
                        curl_easy_setopt(curl, CURLOPT_URL, msg);
                        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
#ifdef HTTP_ENABLE_RESP
                        /* Perform the request, res will get the return code */
                        res = curl_easy_perform(curl);
                        /* Check for errors */
                        if (res != CURLE_OK)
                            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                                curl_easy_strerror(res));
#endif
                        /* always cleanup */
                        curl_easy_cleanup(curl);
                    }

                }

                //                if ((sec_int_pre != sec_int) | (send_touchlife_no == 0)) {
                if ((send_touchlife_no_pre != send_touchlife_no)) {
                    if (send_touchlife_no < 4) send_touchlife_no_pre = send_touchlife_no;
                    sec_int_pre = sec_int;
                }
#endif
                //  else sprintf(text_show, " ");
                   /*
                   if (send_touchlife_no < 4) send_touchlife_no_pre = send_touchlife_no;
                   sec_int_pre = sec_int;
                   */
                   ////////////////////////////////////////////////////////////////////////
                if (have_people_act) have_people1 = 1;
                else have_people1 = 0;

                if (!grayImage_GIS.data) {
                    grayImage_GIS = Mat(240, 320, CV_8UC3, Scalar(0));
                }
                //   cvtColor(resize_img, C_grayImage_4X_rise, CV_GRAY2RGB);
#ifndef net_udp                
                if (frame_no_is == 1) {   //bed_head_up
                    sprintf(text, "%d", frame_no);
                    putText(C_grayImage_4X_rise, text, Point(50, 50), fontFace, 0.8, Scalar(255, 128, 0), thickness, 1);
                }
                cvtColor(resize_img_ir, C_resize_img_ir, CV_GRAY2RGB);
                if (is_engineer) {
                    Mat image_last_rise;
                    resize(image_last, image_last_rise, Size(image.cols * 4, image.rows * 4), 0, 0, INTER_LINEAR);
                    cvtColor(image_last_rise, C_image_last_rise, CV_GRAY2RGB);
                    image_last_rise.release();
                }
#ifdef SHOW_BED_TOF
                if ((frame_no > 2)) {
                    if (no_bed == 0) {
                        rectangle(C_grayImage_4X_rise, Point((col_first_line * 4), (bed_first_line * 4)),
                            Point((col_end_line * 4), (bed_end_line * 4)), Scalar(255, 0, 0), 1, 1, 0);

                    }

                }
#endif
#endif

#ifdef net_udp                
                image_send.copyTo(show_image_original(Rect(0, 0, 80, 60))); //orignal
                image_ir_send.copyTo(show_image_original(Rect(80, 0, 80, 60)));
                // imshow("test image", show_image_original);
                waitKey(1);
#else
                if ((frame_no > 2) && is_engineer) {
#ifndef   SHOW_BED_TOF   

                    rectangle(C_grayImage_4X_rise, Point((col_first_line * 4), (bed_first_line * 4)),
                        Point((col_end_line * 4), (bed_end_line * 4)), Scalar(255, 0, 0), 1, 1, 0);
#endif
                    circle(C_grayImage_4X_rise, Point((now_4x), (now_4y)), 1, Scalar(255, 10, 10), 2, 8, 0);
                    ////////////////////////////////////////////////////////////////////////////////////////
                    if (inside_bed_tmp && bed_obj_status) { //20220126
                        rectangle(C_grayImage_4X_rise, Point((bed_area.start_x * 4), (bed_area.start_y * 4)), //red line
                            Point((bed_area.end_x * 4), (bed_area.end_y * 4)), Scalar(0, 0, 255), 1, 1, 0);

                        line(C_image_last_rise, Point((bed_area.start_x * 4), (bed_area.hight_div_2 * 4)),
                            Point((bed_area.end_x * 4), (bed_area.hight_div_2 * 4)), Scalar(0, 255, 0), 1, 1, 0);
                        line(C_image_last_rise, Point((bed_area.width_div_2 * 4), (bed_area.start_y * 4)),
                            Point((bed_area.width_div_2 * 4), (bed_area.end_y * 4)), Scalar(0, 255, 0), 1, 1, 0);
                        rectangle(C_image_last_rise, Point((col_first_line * 4), (bed_first_line * 4)),
                            Point((col_end_line * 4), (bed_end_line * 4)), Scalar(255, 255, 0), 1, 1, 0);
                        rectangle(C_image_last_rise, Point((bed_area.start_x * 4), (bed_area.start_y * 4)),
                            Point((bed_area.end_x * 4), (bed_area.end_y * 4)), Scalar(0, 255, 255), 1, 1, 0);
                    }
                }
                line(C_grayImage_XX_rise, Point(0, 0), Point(320, 0), Scalar(0, 0, 255), 1, 1, 0);
                line(C_resize_img_ir, Point(0, 0), Point(0, 240), Scalar(0, 0, 255), 1, 1, 0);
                line(grayImage_GIS, Point(0, 0), Point(320, 0), Scalar(0, 0, 255), 1, 1, 0);
                line(grayImage_GIS, Point(0, 0), Point(0, 240), Scalar(0, 0, 255), 1, 1, 0);

                C_grayImage_4X_rise.copyTo(show_image(Rect(0, 0, 320, 240))); //orignal
                C_resize_img_ir.copyTo(show_image(Rect(320, 0, 320, 240)));
                C_grayImage_XX_rise.copyTo(show_image(Rect(0, 240, 320, 240)));
                if (is_engineer) C_image_last_rise.copyTo(show_image(Rect(320, 240, 320, 240)));
                else grayImage_GIS.copyTo(show_image(Rect(320, 240, 320, 240)));
                //    imshow(gis_filename, show_image);
#endif                	

#ifdef RESULT_VIDEO
                //                if ( ((frame_no % 3) != 0))
                writer << show_image;
#endif
                fall_true = 0;

                grayImage_pre_pre = grayImage_pre.clone();
                grayImage_pre = grayImage_now_th.clone();
                pre_x = now_x;
                pre_y = now_y;
                tmpMin = tmpNowMin;
                bed_area_pre = bed_area;
                people_height_pre2 = people_height_pre;
                people_height_pre = people_height;
                no_people_pre = no_people;
                turn_up_pre = turn_up;
            } //if (frame_no)

            //
            frame_no = frame_no + 1;
#ifdef CAMERA_MODE
        } //go_active
    } //new frame
#endif

#ifdef net_udp
                //////////
  //#ifndef CAMERA_MODE
             //   if (IS_NEW == 1) {
    time(&now);
    t = localtime(&now);
    //char* year, * date, time*;
    year_int = (t->tm_year + 1900);
    month_int = (t->tm_mon + 1);
    day_int = (t->tm_mday);
    hour_int = t->tm_hour;
    min_int = t->tm_min;
    year_s = to_string(year_int);
    month_s = to_string(month_int);
    if (month_int < 10)  month_s = "0" + month_s;
    day_s = to_string(day_int);
    if (day_int < 10) day_s = "0" + day_s;

    hour_int = t->tm_hour;
    min_int = t->tm_min;
    sec_int = t->tm_sec;
    hour_s = to_string(hour_int);
    if (hour_int < 10) hour_s = "0" + hour_s;
    min_s = to_string(min_int);
    if (min_int < 10) min_s = "0" + min_s;
    sec_s = to_string(sec_int);
    if (sec_int < 10) sec_s = "0" + sec_s;

    strcpy(date_name, &year_s[0]);
    strcat(date_name, &month_s[0]);
    strcat(date_name, &day_s[0]);

    strcpy(time_name, &hour_s[0]);
    strcat(time_name, &min_s[0]);
    strcat(time_name, &sec_s[0]);
    //  }
//#endif
             //////
    const char* msg_send = "%s%sT%s%s";
#ifdef CAMERA_MODE
    if (frame_no < 5) {
        if (no_bed) sprintf(buffer_net, msg_send, "CameraNoBD", date_name, time_name, "no");
        else sprintf(buffer_net, msg_send, "CameraYesD", date_name, time_name, "yes");
    }
    else sprintf(buffer_net, msg_send, "HugoCamera", date_name, time_name, text_show);
#else                
    if (frame_no < 5) {
        if (no_bed) sprintf(buffer_net, msg_send, "HugoXBegin", date_name, time_name, "no");
        else sprintf(buffer_net, msg_send, "HugoXBegin", date_name, time_name, "yes");
        //   cout << "frame_no : " << frame_no << "_" << no_bed << "_" << date_name << "_" << time_name << "_" << text_show << "\n";
    }
    else sprintf(buffer_net, msg_send, "HugoXBegin", date_name, time_name, text_show);
#endif                	
    int len = strlen(buffer_net) + 1;
    //cout << "send data \n";

 // sprintf(text_show, "Falling");
    //sendto header
    iResult = sendto(sockfd, buffer_net, len, 0, (struct sockaddr*)&server, addr_len);

    //printf("Send head %s %d\n", buffer_net, len);
   /*
        std::vector<uchar> data_encode;
        bool image_encode = imencode(".jpg", show_image_down, data_encode);
        std::string str_encode(data_encode.begin(), data_encode.end());
     */
    usleep(100);
    if (iResult > 0) {
        iResult = 0;
        int cnt = 0;
        char msg[200]; // = "Hello World !\n";

     //   for (int i = 0; i < Send_number; i++) {
                 //send data
        iResult = sendto(sockfd, (char*)show_image_original.data, 9600 * 3, 0,
            (struct sockaddr*)&server, addr_len);
        if (iResult < 0) cout << "The send data loss" << frame_no << "\n";
        /*
        if (!strcmp(text_show, "Falling")) {
                const char* fall_send = "%2d%2d%2d%2d";
                sprintf(buffer_net, fall_send, global_fall_x, global_fall_y, global_fall_width, global_fall_height);
                iResult = sendto(sockfd, (char *)buffer_net , 8, 0,
                                 (struct sockaddr *)&server, addr_len);
        }
        */
        usleep(10);

        ////////////////////////////////////////////////////////////////////////
        cnt = recvfrom(sockfd, msg, 200, 0,
            (struct sockaddr*)&server, &addr_len);
        if (cnt < 0) {
            cout << "rece command err " << frame_no << endl;
            // break;
        }
        usleep(10);
#ifdef CAMERA_MODE

        if (!strcmp(msg, "cmd=g"))go_active = 1;
#endif
        if (!strcmp(msg, "frame_no_is"))frame_no_is = !frame_no_is;
        else if (!strcmp(msg, "display over")) {
            game_bed_fall_over = 1;
            cout << "The end frame " << frame_no << " The over " << game_bed_fall_over << "\n";
        }

        //  } //for Send_number

#ifdef CAMERA_MODE
        if (go_active & !go_active_pre) {
            go_active_pre = 1;
            frame_no = 0;
            cout << "go_active\n";
        }
#endif
        cout << "send over\n";
    }  // if (iResult > 0)
#else
#ifdef IP_CAMERA
        /*
               vector<int> param= vector<int>(2);
               param[0]=CV_IMWRITE_JPEG_QUALITY;
               param[1]=95;//default(95) 0-100
               */
               // cout << "IP_CAMERA\n";
    vector <uchar> data_encode;
    //buffer.resize(MB);

    bool image_encode = imencode(".jpg", show_image, data_encode);
    int buffer_size = data_encode.size();
    const char* msg_sendx = "%d";
    const char* msg_sendy = " %d";
#ifdef DEBUG_PRINT                
    printf("frame size is %d\n", buffer_size);
#endif                
    char buffer_len[6] = "\n";
    char msg[200];
    //char data_send[MB];
    //const char* msg_send_data = "%s";
    //const char* send_data_len = "%Ls";
    bool digit4 = 0;
    if (buffer_size < 10000) {
        sprintf(buffer_len, msg_sendy, buffer_size);
        digit4 = 1;
    }
    else if ((buffer_size > 99999) || (buffer_size < 1000)) {
        printf("The data len error\n");
        exit(1);
    }
    else {
        sprintf(buffer_len, msg_sendx, buffer_size);
    }
#ifdef DEBUG_PRINT                
    printf("The string1 %s\n", buffer_len);
#endif                
    //std::string encode_len = buffer_len;
    //buffer_len[5] = "\n";
    //printf("The string2 %s\n", encode_len.data());
    //std::wstring w_encode_len = s2ws(encode_len);
    //printf("The utf-8 %Ls and len %d\n", w_encode_len.data(),w_encode_len.size());
/*   utf-8
                send(sockfd, w_encode_len.data(), w_encode_len.size()*4, 0);
                int encode_len_tmp = strlen(buffer_len);
                int encode_len_tmp = w_encode_len.size()*4;
                sprintf(data_send, send_data_len, w_encode_len.data());
                send(sockfd, w_encode_len.data(), 20, 0);
                send(sockfd,data_encode.data(), buffer_size, 0);
                string show_status(text_show);
                show_status.data() = text_show;
                std::wstring w_show_status = s2ws(show_status);
                send(sockfd, w_show_status , 10, 0);
*/
//direct send jpg data
    send(sockfd, buffer_len, 5, 0);
    //                vector<uchar> number_x(5); // you are using C++ not C
    char number_x[5]; // you are using C++ not C
    bool loop_tmp = true;
    do {
        // number_x[5]="\n";
        recv(sockfd, number_x, 5, 0);
#ifdef DEBUG_PRINT                      
        cout << "The receive length: " << number_x << " & " << buffer_len << "\n";
#endif                   
        loop_tmp = !digit4 & ((number_x[0] > 0x39) || (number_x[0] < 0x30)) ||
            (number_x[1] > 0x39) || (number_x[1] < 0x30) ||
            (number_x[2] > 0x39) || (number_x[2] < 0x30) ||
            (number_x[3] > 0x39) || (number_x[3] < 0x30) ||
            (number_x[4] > 0x39) || (number_x[4] < 0x30);
        if (loop_tmp == true)  send(sockfd, buffer_len, 5, 0);

    } while (loop_tmp);

    //send jpeg encode data        
    send(sockfd, data_encode.data(), buffer_size, 0);

    //send orignal data             
    if (frame_no < 2) {
        send(sockfd, tof_serial, 12, 0);
    }
    else if ((go_active == 0) & (go_init == 0)) {
        send(sockfd, "no run      ", 12, 0);
    }
    else if ((go_active == 0) & (go_init == 1)) {
        send(sockfd, "init        ", 12, 0);
    }
    else {
        send(sockfd, text_show, 12, 0);
    }
#ifdef DEBUG_PRINT                
    printf("--------------------The frame_no: %d\n", frame_no);
#endif                
    // printf("The mode is %d and %d\n", go_active, go_init);
    vector<uchar> buf(10); // you are using C++ not C
    int bytes = recv(sockfd, buf.data(), buf.size(), 0);
    //std::cout << buf.data() << "\n";   go_init
    if ((go_active == 0) & (strcmp((char*)buf.data(), "Go        ") == 0))
    {
        frame_no = 0;
        go_init = 0;
        go_active = 1;
        printf("The active processing\n");
    }
    else if ((go_active == 1) & (strcmp((char*)buf.data(), "Ack       ") == 0))
    {
        go_active = 0;
        go_init = 0;
        frame_no = 0;
        cout << "The end frame " << frame_no << " \n";
    }
    else if ((go_init == 0) & (strcmp((char*)buf.data(), "init      ") == 0))
    {
        go_init = 1;
        go_active = 0;
        frame_no = 0;
        printf("The initial processing\n");
    }
    // printf("The mode is %d and %d\n", go_active, go_init);
    //20220124 modify by hugoliu


#else               
    imshow(gis_filename, show_image);
#endif                    
#endif
    // sprintf(text_show, " ");
//             }  // if (iResult > 0)
    other_people_inside_pre = other_people_inside;
    center_area_start_x = 0;
    center_area_start_y = 0;
    center_area_end_x = 0;
    center_area_end_y = 0;
    fall_area_start_x = 0;
    fall_area_start_y = 0;
    fall_area_end_x = 0;
    fall_area_end_y = 0;
    //text_show = "";
#ifdef CAMERA_MODE
    iWaitKey = waitKey(1) & 0xFF;
    if (iWaitKey == 'g') {
        frame_no = 0;
        go_active = true;
    }
    if (iWaitKey == 'q') {
        frame_no = 0;
        cout << "The end frame " << frame_no << " \n";
    }
#else
    iKey = waitKey(stoi(argv[2]));
    if ((iKey == 27) || (iKey == 'q')) {
        game_bed_fall_over = 1;
    }
#ifdef SAVE_JPG
    else if (frame_no > 4) {
        imwrite(image_name, C_grayImage_4X_rise);
        game_bed_fall_over = 1;
    }
#endif
    //sotp_frame
    else if ((iKey == 's'))
        //           else if ((iKey == 's') || (((frame_no > 171) && (frame_no < 176)) || (frame_no > 271))) //272
        //         else if ((iKey == 's') ||  (frame_no > 368)) //272
    {
        namedWindow("stopshow", WINDOW_AUTOSIZE);
        imshow("stopshow", C_grayImage_4X_rise);

        printf("play stop at %d\n", frame_no);
        setMouseCallback("stopshow", mouse_callback);
        int xKey;
        do {
            xKey = cv::waitKey(1);
            if (xKey == 'q') {
                game_bed_fall_over = 1;
            }
        } while ((xKey != 'g') && (!game_bed_fall_over));
        // local_img.release();
        destroyWindow("stopshow");

        if (!game_bed_fall_over) cout << "play go" << endl;
    }
    else if (iKey == 'e') is_engineer = 1;
    else if (iKey == 'n') is_engineer = 0;
    if (game_bed_fall_over) result = 0;
    else {
        int i = 0;
        do {

            result = fread(buffer_head, sizeof(unsigned char), 8, fr);
            result = fread(bufferX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr);
            fread(buffer_head_ir, sizeof(unsigned char), 8, fr_ir);
            result_ir = fread(buffer_irX, sizeof(unsigned char), TOF_DEPTH_PIXELS, fr_ir);

            i++;
        } while ((i < (draw_frame + 1)) & (result > 0) & (result_ir > 0));
    }
            } while (result > 0);

#ifdef Compare_LOG
            if (result < 1) game_bed_fall_over = 1;
#endif
#ifdef VIDEO_RECORD
            if (result < 1) game_bed_fall_over = 1;
#endif
#ifdef NO_REPEAT
            if (result < 1) game_bed_fall_over = 1;
#endif
#endif //cmaera mode
        } //while(game_bed_fall_over) or (cmaera)
        show_image.release();
        image.release();
        Image_mask.release();
        image_last.release();
        depth8U.release();
        image_ir.release();
        depth8U_ir.release();
        //    grayImage_XX_rise.release();
        C_grayImage_XX_rise.release();
        grayImage_GIS.release();
        image_pre_fall.release();
        grayImage_now.release();
        grayImage_now_th.release();
        grayImage_pre.release();;
        grayImage_init.release();
        grayImage_initX.release();
        //    now_image_area.release();
#ifndef net_udp    
        resize_img_ir.release();
        C_resize_img_ir.release();
        C_grayImage_4X_rise.release();
        C_image_last_rise.release();
        resize_img.release();
#endif    
        //destroyWindow("resizex4");
#ifndef net_udp
        destroyWindow(gis_filename); // CV_WINDOW_AUTOSIZE);
        destroyWindow("controller");
#endif
#ifdef DEBUG_X_FIG
        cvDestroyWindow("diff");
        cvDestroyWindow("fall_area");
#endif
#ifdef VIDEO_RECORD
        writer_source.release(); // cimage;
        writer_ir.release();// cimage_ir;
#ifdef RESULT_VIDEO
        writer.release();
#endif
#endif
#ifdef Compare_LOG
        fclose(ComPfw_log);
#endif
#ifdef IMAGE_MASK
        destroyWindow("mask");
#endif
#ifdef IMAGE_INIT
        cvDestroyWindow("image_init");
#endif
#ifndef CAMERA_MODE
        free(bufferX);
        //    free(buffer_head);
        free(buffer_irX);
        free(buffer_head_ir);
#endif
#ifdef CAMERA_MODE
# ifdef SAVE_DATA
        if (go_active) {
            fclose(fw);
            fclose(fw_ir);
        }
#endif    
        stop_capturing();
        uninit_uvc_device();

        /*
         * Stop device
         */
        close_uvc_device();

        // goto reconnect;
    } while (wotking_camera);
#else
        fclose(fr);
        fclose(fr_ir);
#ifdef LOG_FILE
        fclose(fw_log);
#endif
#endif
        /*
        closesocket(sockClient);//關閉連線
        WSACleanup();
        */
        return 0;
    }


    /*
     int mac_address() {
             int fd;
             struct ifreq ifr;
             char *iface = "eth0";
             unsigned char *mac;

             fd = socket(AF_INET, SOCK_DGRAM, 0);

             ifr.ifr_addr.sa_family = AF_INET;
             strncpy(ifr.ifr_name , iface , IFNAMSIZ-1);

             ioctl(fd, SIOCGIFHWADDR, &ifr);

             close(fd);

             mac = (unsigned char *)ifr.ifr_hwaddr.sa_data;

             //display mac address
             printf("Mac : %.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n" , mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

             return 0;
         }
    */
