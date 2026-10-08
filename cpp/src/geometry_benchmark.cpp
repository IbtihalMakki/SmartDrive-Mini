#include <string>
#include "rgbd_geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        const int repetitions = argc > 1 ? std::stoi(argv[1]) : 2000;
        const int warmup = argc > 2 ? std::stoi(argv[2]) : 200;
        if (repetitions < 20 || warmup < 0 || repetitions > 1000000) throw std::invalid_argument("Invalid iterations");
        constexpr int width = 640, height = 480;
        std::vector<double> depth(static_cast<std::size_t>(width) * height);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                depth[static_cast<std::size_t>(y)*width+x] = (x+y)%17 == 0 ? 0.0 : 8.0 + ((x*3+y*7)%101)*0.01;
        const auto k = smartdrive::camera_intrinsics(width, height, 90.0);
        const smartdrive::Box box{200, 120, 440, 360};
        for (int i = 0; i < warmup; ++i) (void)smartdrive::localize_visible_surface(depth,width,height,box,k);
        std::vector<double> samples;
        samples.reserve(repetitions);
        smartdrive::Localization last;
        for (int i = 0; i < repetitions; ++i) {
            const auto start = std::chrono::steady_clock::now();
            last = smartdrive::localize_visible_surface(depth,width,height,box,k);
            const auto end = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double,std::milli>(end-start).count());
        }
        if (!last.valid) throw std::runtime_error("Localization failed");
        std::sort(samples.begin(), samples.end());
        double sum=0;
        for (auto s : samples) sum+=s;
        const double mean=sum/samples.size();
        const std::size_t idx=static_cast<std::size_t>(std::ceil(0.95*samples.size()))-1;
        std::cout << std::setprecision(15)
                  << "{\"language\":\"cpp17\",\"iterations\":" << repetitions
                  << ",\"warmup\":" << warmup
                  << ",\"mean_ms\":" << mean
                  << ",\"p95_ms\":" << samples[idx]
                  << ",\"throughput_ops_s\":" << 1000.0/mean
                  << ",\"position\":[" << last.point.x << "," << last.point.y << "," << last.point.z << "]"
                  << ",\"sample_count\":" << last.sample_count << "}\n";
        return 0;
    } catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
