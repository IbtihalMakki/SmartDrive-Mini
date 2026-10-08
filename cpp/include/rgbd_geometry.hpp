#pragma once
#include <cstddef>
#include <vector>

namespace smartdrive {
struct Intrinsics { double fx, fy, cx, cy; };
struct Point3D { double x, y, z; };
struct Box { double x1, y1, x2, y2; };
struct Localization { bool valid = false; Point3D point{0,0,0}; std::size_t sample_count = 0; };

Intrinsics camera_intrinsics(int width, int height, double horizontal_fov_deg);
Point3D back_project(double u, double v, double depth_m, const Intrinsics& k);
Localization localize_visible_surface(const std::vector<double>& depth, int width, int height,
                                      const Box& box, const Intrinsics& k);
}
