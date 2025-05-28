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

#ifndef HORUS_LIB_ANALYZE_BASE_HPP_
#define HORUS_LIB_ANALYZE_BASE_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <chrono>
#include <queue>
#include <cmath>
#include <filesystem>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "lib_horus_type.hpp"

// CSVファイルをmmapで読み取り、ヘッダーとデータをパースして返す
static std::pair<std::vector<std::string>, str_data_base>
parse_csv_mmap(const std::string& filepath)
{
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Can not open file: " + filepath);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        throw std::runtime_error("fstat failed: " + filepath);
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    void* file_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed: " + filepath);
    }

    close(fd);

    auto* char_data = static_cast<const char*>(file_data);

    std::vector<size_t> line_starts; 
    line_starts.push_back(0);

    for (size_t i = 0; i < file_size; ++i) {
        if (char_data[i] == '\n') {
            if (i + 1 < file_size) {
                line_starts.push_back(i + 1);
            }
        }
    }

    std::vector<std::string> header;
    {
        size_t start_pos = line_starts[0];
        size_t end_pos   = (line_starts.size() > 1) ? line_starts[1] : file_size;

        std::string header_line(&char_data[start_pos], end_pos - start_pos);
        if (!header_line.empty() && header_line.back() == '\n') {
            header_line.pop_back();
        }

        std::stringstream ss(header_line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            header.push_back(cell);
        }
    }

    const size_t data_line_count = (line_starts.size() > 1)
                                   ? (line_starts.size() - 1)
                                   : 0;

    str_data_base cells;
    cells.resize(data_line_count);

    #pragma omp parallel for schedule(static)
    for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(data_line_count); ++idx) {
        size_t start_pos = line_starts[idx + 1];
        size_t end_pos   = (idx + 2 < line_starts.size())
                           ? line_starts[idx + 2]
                           : file_size;

        std::string line(&char_data[start_pos], end_pos - start_pos);
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        row.reserve(16);

        while (std::getline(ss, cell, ',')) {
            row.emplace_back(std::move(cell));
        }

        cells[idx] = std::move(row);
    }

    munmap(file_data, file_size);

    return std::make_pair(header, cells);
}

// 中心座標と幅・高さから矩形の5点（中心、4隅）を生成する
auto xyhw_to_format(uint16_t x, uint16_t y, uint16_t h, uint16_t w) -> horus_data_format
{
    uint16_t half_width  = w / 2;
    uint16_t half_height = h / 2;

    point center(x, y);
    point top_right(x + half_width,  y + half_height);
    point bottom_right(x + half_width,  y - half_height);
    point top_left(x - half_width,  y + half_height);
    point bottom_left(x - half_width,  y - half_height);

    return horus_data_format(center, top_right, bottom_right, top_left, bottom_left);
}

// 高速な文字列から整数への変換（uint16_t想定）
inline auto fast_atoi(const std::string & input) -> uint16_t
{
    const char * str = input.c_str();
    int val = 0;
    while( *str ) {
        val = val*10 + (*str++ - '0');
    }
    return val;
}

// CSV由来の文字列データを内部形式（map）に変換する
auto convert_to_horus_format(
    const str_data_base& data
) -> horus_bin_format
{
    std::vector<std::vector<std::tuple<int,int,horus_data_format>>> local_temp(omp_get_max_threads());

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& vec_for_this_thread = local_temp[thread_id];
        vec_for_this_thread.reserve(data.size() / omp_get_num_threads() + 1);

        #pragma omp for schedule(static)
        for (std::int64_t i = 0; i < (std::int64_t)data.size(); ++i) {
            const auto& row = data[i];
            if (row.size() != 6) {
                continue;
            }

            uint16_t inner_key = fast_atoi(row[0]);
            uint16_t outer_key = fast_atoi(row[1]);

            uint16_t x_center = fast_atoi(row[2]);
            uint16_t y_center = fast_atoi(row[3]);
            uint16_t w        = fast_atoi(row[4]);
            uint16_t h        = fast_atoi(row[5]);

            auto fmt = xyhw_to_format(x_center, y_center, w, h);
            vec_for_this_thread.emplace_back(outer_key, inner_key, fmt);
        }
    }

    horus_bin_format results;
    for (auto &thread_vec : local_temp) {
        for (auto &elem : thread_vec) {
            int outer_key = std::get<0>(elem);
            int inner_key = std::get<1>(elem);
            auto& format  = std::get<2>(elem);
            results[outer_key][inner_key] = format;
        }
    }

    return results;
}

// 座標のノルム（二乗距離）を計算する
auto norm(const point & p) -> uint16_t
{
    auto result = p.x * p.x + p.y * p.y;
    return result;
}

// horus_data_formatのマップに対して移動平均を適用する
std::map<int, horus_data_format> moving_average(
    const std::map<int, horus_data_format>& data_container, int avg_size) 
{
    if (avg_size <= 0) {
        throw std::invalid_argument("avg_size must be greater than 0");
    }

    std::map<int, horus_data_format> result;
    horus_data_format cumulative_sum{};
    int count = 0;

    auto it = data_container.begin();
    std::deque<std::pair<int, horus_data_format>> window;

    while (it != data_container.end()) {
        const auto& [key, data] = *it;

        cumulative_sum += data;

        window.push_back({key, data});
        count++;

        if (count > avg_size) {
            const auto& [old_key, old_data] = window.front();
            cumulative_sum -= old_data;

            window.pop_front();
            count--;
        }
        result[key] = cumulative_sum / count;

        ++it;
    }

    return result;
}

#endif // HORUS_LIB_ANALYZE_BASE_HPP_
