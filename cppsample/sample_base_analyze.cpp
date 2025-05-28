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
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

// --- OpenMP ---
#include <omp.h>

// --- POSIX (システム依存のヘッダ) ---
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- Project Headers ---
#include "lib_horus_type.hpp"
#include "lib_analyze_base.hpp"


// horus_analyze_formatのベクターに対して移動平均を適用する
std::vector<horus_analyze_format> moving_average(
    const std::vector<horus_analyze_format>& data_container, int avg_size) 
{
    if (avg_size <= 0) {
        throw std::invalid_argument("avg_size must be greater than 0");
    }

    std::vector<horus_analyze_format> result;
    horus_analyze_format cumulative_sum{};
    int count = 0;

    auto it = data_container.begin();
    std::deque<horus_analyze_format> window;

    for(const auto & data : data_container)
    {
        cumulative_sum += data;
        window.push_back(data);
        count++;

        if (count > avg_size) {
            const auto& old_data = window.front();
            cumulative_sum -= old_data;

            window.pop_front();
            count--;
        }
        result.push_back(cumulative_sum / count);

    }

    return result;
}

// horus_data_formatの変化量を計算し、平均を取って分析用形式に変換する
auto analyze_avgcmp_delta(const std::map<int, horus_data_format>& data_container, int avg_size) 
{
    auto data_container_ = moving_average(data_container, 30);
    std::vector<horus_analyze_format> internal_data;
    auto prev_data = horus_data_format();
    for (const auto & [key, data] : data_container_) 
    {
        const auto delta = data - prev_data;
        internal_data.push_back(horus_analyze_format(
            norm(delta.center),
            norm(delta.top_right),
            norm(delta.bottom_right),
            norm(delta.top_left),
            norm(delta.bottom_left)
        ));

        prev_data = data;
    }

    int count = 0;
    horus_analyze_format avg_data;
    std::vector<horus_analyze_format> result_data;
    for (const auto & data : internal_data) 
    {
        avg_data += data;
        count++;
        if (count == avg_size) 
        {
            result_data.push_back(avg_data);
            avg_data = horus_analyze_format();
            count = 0;
        }
    }
    result_data.push_back(avg_data);

    result_data.erase(result_data.begin());
    result_data = moving_average(result_data, 10);

    return result_data;
}

// horus_analyze_formatのデータをCSV形式でファイルに書き出す
void write_data_to_csv(const std::vector<horus_analyze_format> & data_container,
                                const std::string & filename)
{
    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp) 
    {
        std::perror(("Failed to open the file: " + filename).c_str());
        return;
    }

    std::fputs("norm_center,"
               "norm_bottom_left,"
               "norm_bottom_right,"
               "norm_top_left,"
               "norm_top_right,\n", fp);


    char lineBuffer[1024];

    for (auto const & data : data_container) 
    {
        int len = std::sprintf(
            lineBuffer,
            "%d,%d,%d,%d,%d\n",
            data.norm_center,
            data.norm_bottom_left,
            data.norm_bottom_right,
            data.norm_top_left,
            data.norm_top_right
        );
        std::fwrite(lineBuffer, 1, len, fp);
    }

    std::fclose(fp);
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << argc << std::endl;
        std::cerr << "usage: ./horuscpp [InputCSVFile] [OutputDirectry]" << std::endl;
        return 1;
    }

    const std::string input_filename = argv[1];
    const std::string output_dir = argv[2];

    std::filesystem::create_directories(output_dir);

    const auto start_time = std::chrono::high_resolution_clock::now();

    const auto [_, str_data] = parse_csv_mmap(input_filename);

    auto horus_data = convert_to_horus_format(str_data);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(horus_data.size()); ++i) {
        auto it = std::next(horus_data.begin(), i);
        const auto& [cid, data] = *it;

        #pragma omp critical
        std::cout << "class id=" << cid << " elements size=" << data.size() << std::endl;
        const auto alyzed_data = analyze_avgcmp_delta(data, 150);
        const auto filename = output_dir + "/analyzed_" + std::to_string(cid) + ".csv";
        write_data_to_csv(alyzed_data, filename);
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    auto millisec = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) / 1000.0;
    std::cout << (millisec / 1000.0) << " sec" << std::endl;

    return 0;
}

// コンパイル方法と実行方法
// g++ -std=c++17 -fopenmp -o horuscpp sample_base_analyze.cpp
// ./horuscpp CSV_FILE_PATH OUTPUT_DIR
