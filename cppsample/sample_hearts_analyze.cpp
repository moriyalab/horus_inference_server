// Copyright 2024 Taiga Takano
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// --- C++ Standard Library ---
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// --- OpenCV ---
#include <opencv2/opencv.hpp>

// --- Project Headers ---
#include "lib_horus_type.hpp"
#include "lib_analyze_base.hpp"

/**
 * @brief 動画中の特定 ID の bbox を切り出し、面積・輝度を CSV に書き出す
 */
void analyze_bbox_video(
    const std::string& video_path,
    const std::map<int, horus_data_format>& bbox_data,
    const std::string& output_csv)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("動画が開けませんでした: " + video_path);
    }

    std::ofstream ofs(output_csv);
    if (!ofs) {
        throw std::runtime_error("CSV ファイルが開けませんでした: " + output_csv);
    }
    ofs << "frame,area,brightness\n";
    ofs.flush();

    cv::Mat frame;
    int frame_idx = 0;
    while (cap.read(frame)) {
        ++frame_idx;
        if (bbox_data.find(frame_idx - 1) == bbox_data.end()) continue;
        const auto frame_bbox = bbox_data.at(frame_idx - 1);

        int x1 = frame_bbox.top_left.x;
        int y1 = frame_bbox.top_left.y;
        int x2 = frame_bbox.bottom_right.x;
        int y2 = frame_bbox.bottom_right.y;

        x1 = std::clamp(x1, 0, frame.cols - 1);
        x2 = std::clamp(x2, 0, frame.cols - 1);
        y1 = std::clamp(y1, 0, frame.rows - 1);
        y2 = std::clamp(y2, 0, frame.rows - 1);

        int w = std::abs(x2 - x1);
        int h = std::abs(y2 - y1);

        if (w > 0 and h > 0){
            cv::Mat patch = frame(cv::Rect(x1, y1, w, h));
            int area = w * h;
            cv::Mat gray;
            cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
            double brightness = cv::mean(gray)[0] / 255.0;
            ofs << frame_idx << "," << area << "," << brightness << "\n";
            ofs.flush();
        }
    }
    cap.release();
    ofs.close();
}

int main(int argc, char* argv[])
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    // 引数: [1]=BBox CSV, [2]=動画ファイル, [3]=ターゲットID, [4]=出力CSV
    if (argc != 5) {
        std::cerr << "usage: " << argv[0]
                  << " [InputBBoxCSV] [VideoFile] [TargetID] [OutputCSV]\n";
        return 1;
    }
    const std::string input_csv   = argv[1];
    const std::string video_file  = argv[2];
    const std::string id_str      = argv[3];
    const std::string output_csv  = argv[4];

    int target_id;
    try {
        target_id = std::stoi(id_str);
    } catch (const std::exception&) {
        std::cerr << "Invalid TargetID: " << id_str << "\n";
        return 1;
    }

    auto [header, str_data] = parse_csv_mmap(input_csv);
    auto bbox_data = convert_to_horus_format(str_data).at(target_id);

    // 解析開始
    try {
        analyze_bbox_video(video_file, bbox_data, output_csv);
    } catch (const std::exception& e) {
        std::cerr << "解析中にエラー: " << e.what() << "\n";
        return 1;
    }
    std::cout << bbox_data.size() << std::endl;;

    const auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "解析完了 (所要時間: " << elapsed << " sec)\n";

    return 0;
}
