// Include necessary headers
#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // Set CUDA device
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const fs::path path{argv[2]};

    std::vector<std::string> imagePathList;
    bool isVideo{false};

    auto yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov"
                 || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat image;
    cv::Size size = cv::Size{640, 640};
    std::vector<Object> objs;

    // Warmup
    if (!isVideo && !imagePathList.empty()) {
        image = cv::imread(imagePathList[0]);
        yolov8->copy_from_Mat(image, size);
        yolov8->infer();
        yolov8->postprocess(objs);
    }

    // Time inference
    double total_time = 0.0;
    int num_inferences = 100;

    for (int i = 0; i < num_inferences; ++i) {
        if (isVideo) {
            cv::VideoCapture cap(path);

            if (!cap.isOpened()) {
                printf("can not open %s\n", path.c_str());
                return -1;
            }
            while (cap.read(image)) {
                objs.clear();
                yolov8->copy_from_Mat(image, size);
                auto start = std::chrono::high_resolution_clock::now();
                yolov8->infer();
                auto end = std::chrono::high_resolution_clock::now();
                yolov8->postprocess(objs);

                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
        }
        else {
            for (auto& p : imagePathList) {
                objs.clear();
                image = cv::imread(p);
                yolov8->copy_from_Mat(image, size);
                auto start = std::chrono::high_resolution_clock::now();
                yolov8->infer();
                auto end = std::chrono::high_resolution_clock::now();
                yolov8->postprocess(objs);

                total_time += std::chrono::duration<double, std::milli>(end - start).count();
            }
        }
    }

    printf("Average inference time: %.2f ms\n", total_time / num_inferences);

    delete yolov8;
    return 0;
}