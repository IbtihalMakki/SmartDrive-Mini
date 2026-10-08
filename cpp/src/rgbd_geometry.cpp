
#include "rgbd_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace smartdrive {

namespace {

constexpr double PI = 3.14159265358979323846;

bool finite(double value) {
    return std::isfinite(value);
}

void check_intrinsics(const Intrinsics& k) {
    if (!finite(k.fx) ||
        !finite(k.fy) ||
        !finite(k.cx) ||
        !finite(k.cy) ||
        k.fx <= 0.0 ||
        k.fy <= 0.0) {
        throw std::invalid_argument("Invalid camera intrinsics");
    }
}

// Compute median using selection rather than full sorting.
double median(std::vector<double>& values) {
    const std::size_t n = values.size();

    if (n == 0) {
        throw std::invalid_argument("Cannot compute median of empty data");
    }

    const std::size_t middle = n / 2;

    std::nth_element(
        values.begin(),
        values.begin() + middle,
        values.end()
    );

    const double upper = values[middle];

    if (n % 2 != 0) {
        return upper;
    }

    // The maximum element in the lower partition is the
    // lower middle value for an even-length vector.
    const double lower = *std::max_element(
        values.begin(),
        values.begin() + middle
    );

    return (lower + upper) / 2.0;
}

} // namespace


Intrinsics camera_intrinsics(
    int width,
    int height,
    double fov
) {
    if (width <= 0 ||
        height <= 0 ||
        !finite(fov) ||
        fov <= 0.0 ||
        fov >= 180.0) {
        throw std::invalid_argument(
            "Invalid camera dimensions or horizontal FOV"
        );
    }

    const double focal = width / (
        2.0 * std::tan(fov * PI / 360.0)
    );

    return {
        focal,
        focal,
        width / 2.0,
        height / 2.0
    };
}


Point3D back_project(
    double u,
    double v,
    double z,
    const Intrinsics& k
) {
    check_intrinsics(k);

    if (!finite(u) ||
        !finite(v) ||
        !finite(z) ||
        z <= 0.0 ||
        z >= 999.0) {
        throw std::invalid_argument(
            "Invalid pixel or metric depth"
        );
    }

    return {
        (u - k.cx) * z / k.fx,
        (v - k.cy) * z / k.fy,
        z
    };
}


Localization localize_visible_surface(
    const std::vector<double>& depth,
    int width,
    int height,
    const Box& box,
    const Intrinsics& k
) {
    check_intrinsics(k);

    if (width <= 0 ||
        height <= 0 ||
        depth.size() !=
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height)) {
        throw std::invalid_argument(
            "Expected H x W metric depth"
        );
    }

    // Validate bounding box.
    if (!finite(box.x1) ||
        !finite(box.y1) ||
        !finite(box.x2) ||
        !finite(box.y2) ||
        box.x2 <= box.x1 ||
        box.y2 <= box.y1) {
        return {};
    }

    // Central 50% ROI, matching Python implementation.
    const double dx = (box.x2 - box.x1) * 0.25;
    const double dy = (box.y2 - box.y1) * 0.25;

    const int left = static_cast<int>(
        std::clamp(
            std::floor(box.x1 + dx),
            0.0,
            static_cast<double>(width)
        )
    );

    const int right = static_cast<int>(
        std::clamp(
            std::ceil(box.x2 - dx),
            0.0,
            static_cast<double>(width)
        )
    );

    const int top = static_cast<int>(
        std::clamp(
            std::floor(box.y1 + dy),
            0.0,
            static_cast<double>(height)
        )
    );

    const int bottom = static_cast<int>(
        std::clamp(
            std::ceil(box.y2 - dy),
            0.0,
            static_cast<double>(height)
        )
    );

    if (right <= left || bottom <= top) {
        return {};
    }

    const std::size_t roi_area =
        static_cast<std::size_t>(right - left) *
        static_cast<std::size_t>(bottom - top);

    // Preallocate storage to avoid repeated reallocations.
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;

    xs.reserve(roi_area);
    ys.reserve(roi_area);
    zs.reserve(roi_area);

    // Calculate reciprocals once.
    const double inv_fx = 1.0 / k.fx;
    const double inv_fy = 1.0 / k.fy;

    // Extract valid depth and back-project into camera space.
    for (int y = top; y < bottom; ++y) {
        const std::size_t row_offset =
            static_cast<std::size_t>(y) *
            static_cast<std::size_t>(width);

        const double y_factor =
            (static_cast<double>(y) - k.cy) * inv_fy;

        for (int x = left; x < right; ++x) {
            const double z = depth[
                row_offset + static_cast<std::size_t>(x)
            ];

            if (!finite(z) ||
                z <= 0.0 ||
                z >= 999.0) {
                continue;
            }

            const double x_factor =
                (static_cast<double>(x) - k.cx) * inv_fx;

            xs.push_back(x_factor * z);
            ys.push_back(y_factor * z);
            zs.push_back(z);
        }
    }

    const std::size_t count = zs.size();

    // At least 4 valid samples and 50% ROI coverage.
    if (count < 4 || count * 2 < roi_area) {
        return {};
    }

    const Point3D position = {
        median(xs),
        median(ys),
        median(zs)
    };

    return {
        true,
        position,
        count
    };
}

} // namespace smartdrive
