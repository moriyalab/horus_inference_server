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

#ifndef HORUS_TYPE_HPP_
#define HORUS_TYPE_HPP_

#include <map>
#include <vector>
#include <string>


struct point {
    float x;
    float y;
    point() : x(0), y(0) {}
    point(float px, float py) : x(px), y(py) {}
};

auto operator+=(point& p1, const point& p2)
{
    p1.x += p2.x;
    p1.y += p2.y;
}

auto operator/(const point& p1, const float& s)
{
    return point(
        p1.x / s,
        p1.y / s
    );
}

auto operator-(const point& p1, const point& p2)
{
    return point(
        p1.x - p2.x,
        p1.y - p2.y
    );
}

auto operator*(const point& p1, const point& p2)
{
    return point(
        p1.x * p2.x,
        p1.y * p2.y
    );
}

struct horus_data_format {
    point center;
    point top_right;
    point bottom_right;
    point top_left;
    point bottom_left;

    horus_data_format()
        : center(0, 0), top_right(0, 0), bottom_right(0, 0), top_left(0, 0), bottom_left(0, 0) {}
    horus_data_format(point c, point tr, point br, point tl, point bl)
        : center(c), top_right(tr), bottom_right(br), top_left(tl), bottom_left(bl) {}
};

auto operator/(const horus_data_format& d1, const float& s)
{
    return horus_data_format(
        d1.center / s,
        d1.top_right / s,
        d1.bottom_right / s,
        d1.top_left / s,
        d1.bottom_left / s
    );
}

auto operator-(const horus_data_format& d1, const horus_data_format& d2)
{
    return horus_data_format(
        d1.center - d2.center,
        d1.top_right - d2.top_right,
        d1.bottom_right - d2.bottom_right,
        d1.top_left - d2.top_left,
        d1.bottom_left - d2.bottom_left
    );
}

auto operator+=(horus_data_format& d1, const horus_data_format& d2)
{
    d1.center += d2.center;
    d1.top_right += d2.top_right;
    d1.bottom_right += d2.bottom_right;
    d1.top_left += d2.top_left;
    d1.bottom_left += d2.bottom_left;
}

struct horus_analyze_format {
    horus_data_format horus_data;
    float norm_center;
    float norm_top_right;
    float norm_bottom_right;
    float norm_top_left;
    float norm_bottom_left;

    horus_analyze_format()
        : horus_data(horus_data_format()), 
            norm_center(0),
            norm_top_right(0), 
            norm_bottom_right(0), 
            norm_top_left(0), 
            norm_bottom_left(0) {}
    horus_analyze_format(
        horus_data_format hdf,
        float c, float tr, float br, float tl, float bl
    ): horus_data(hdf), 
            norm_center(c),
            norm_top_right(tr), 
            norm_bottom_right(br), 
            norm_top_left(tl), 
            norm_bottom_left(bl) {}
};

using str_data_base = std::vector<std::vector<std::string>>;
using horus_bin_format = std::map<int, std::map<int, horus_data_format>>;

#endif // HORUS_TYPE_HPP_