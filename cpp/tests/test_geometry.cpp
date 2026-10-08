#include "rgbd_geometry.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace smartdrive;
static bool close(double a,double b) { return std::abs(a-b)<1e-10; }
int main() {
    auto k=camera_intrinsics(1280,720,90);
    assert(close(k.fx,640) && close(k.fy,640) && close(k.cx,640) && close(k.cy,360));
    auto p=back_project(704,392,10,k);
    assert(close(p.x,1) && close(p.y,0.5) && close(p.z,10));
    std::vector<double> depth(8*8,10);
    auto r=localize_visible_surface(depth,8,8,{0,0,8,8},camera_intrinsics(8,8,90));
    assert(r.valid && r.sample_count==16);
    assert(close(r.point.x,-1.25) && close(r.point.y,-1.25) && close(r.point.z,10));
    depth[2*8+2]=std::numeric_limits<double>::quiet_NaN();
    depth[2*8+3]=0;
    depth[2*8+4]=999;
    depth[2*8+5]=-1;
    r=localize_visible_surface(depth,8,8,{0,0,8,8},camera_intrinsics(8,8,90));
    assert(r.valid && r.sample_count==12);
    for (int y=2;y<6;y++) for(int x=2;x<6;x++) depth[y*8+x]=0;
    r=localize_visible_surface(depth,8,8,{0,0,8,8},camera_intrinsics(8,8,90));
    assert(!r.valid);
    bool threw=false;
    try { camera_intrinsics(1280,720,180); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
    std::cout << "PASS: C++ intrinsics, projection, ROI median, depth filtering, invalid inputs\n";
}
