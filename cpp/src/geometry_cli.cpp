#include "rgbd_geometry.hpp"
#include <iomanip>
#include <iostream>
int main() {
    using namespace smartdrive;
    auto k=camera_intrinsics(1280,720,90);
    auto p=back_project(704,392,10,k);
    std::vector<double> d(64,10);
    auto r=localize_visible_surface(d,8,8,{0,0,8,8},camera_intrinsics(8,8,90));
    std::cout << std::setprecision(17) << p.x << " " << p.y << " " << p.z << "\n"
              << r.point.x << " " << r.point.y << " " << r.point.z << " " << r.sample_count << "\n";
}
