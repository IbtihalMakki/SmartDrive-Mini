#include "rgbd_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace smartdrive {
namespace {
constexpr double PI = 3.14159265358979323846;
bool finite(double v) { return std::isfinite(v); }
void check_intrinsics(const Intrinsics& k) {
    if (!finite(k.fx) || !finite(k.fy) || !finite(k.cx) || !finite(k.cy) || k.fx <= 0 || k.fy <= 0)
        throw std::invalid_argument("Invalid camera intrinsics");
}
double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    const auto n = v.size();
    return n % 2 ? v[n/2] : (v[n/2 - 1] + v[n/2]) / 2.0;
}
}
Intrinsics camera_intrinsics(int width, int height, double fov) {
    if (width <= 0 || height <= 0 || !finite(fov) || fov <= 0 || fov >= 180)
        throw std::invalid_argument("Invalid camera dimensions or horizontal FOV");
    double f = width / (2.0 * std::tan(fov * PI / 360.0));
    return {f, f, width / 2.0, height / 2.0};
}
Point3D back_project(double u, double v, double z, const Intrinsics& k) {
    check_intrinsics(k);
    if (!finite(u) || !finite(v) || !finite(z) || z <= 0 || z >= 999)
        throw std::invalid_argument("Invalid pixel or metric depth");
    return {(u-k.cx)*z/k.fx, (v-k.cy)*z/k.fy, z};
}
Localization localize_visible_surface(const std::vector<double>& depth, int width, int height,
                                      const Box& box, const Intrinsics& k) {
    check_intrinsics(k);
    if (width <= 0 || height <= 0 || depth.size() != static_cast<std::size_t>(width)*height)
        throw std::invalid_argument("Expected H x W metric depth");
    if (!finite(box.x1) || !finite(box.y1) || !finite(box.x2) || !finite(box.y2)
        || box.x2 <= box.x1 || box.y2 <= box.y1) return {};
    const double dx=(box.x2-box.x1)*0.25, dy=(box.y2-box.y1)*0.25;
    const int left=static_cast<int>(std::clamp(std::floor(box.x1+dx),0.0,static_cast<double>(width)));
    const int right=static_cast<int>(std::clamp(std::ceil(box.x2-dx),0.0,static_cast<double>(width)));
    const int top=static_cast<int>(std::clamp(std::floor(box.y1+dy),0.0,static_cast<double>(height)));
    const int bottom=static_cast<int>(std::clamp(std::ceil(box.y2-dy),0.0,static_cast<double>(height)));
    if (right <= left || bottom <= top) return {};
    std::vector<double> xs,ys,zs;
    for (int y=top; y<bottom; ++y) for (int x=left; x<right; ++x) {
        double z=depth[static_cast<std::size_t>(y)*width+x];
        if (!finite(z) || z <= 0 || z >= 999) continue;
        auto p=back_project(x,y,z,k);
        xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z);
    }
    std::size_t n=zs.size(), area=static_cast<std::size_t>(right-left)*(bottom-top);
    if (n < 4 || n*2 < area) return {};
    return {true,{median(xs),median(ys),median(zs)},n};
}
}
