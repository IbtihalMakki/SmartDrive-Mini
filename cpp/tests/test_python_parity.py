"""Cross-check C++ synthetic fixture output against Python reference equations."""
import math
import subprocess
import sys


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main(exe):
    lines = subprocess.check_output([exe], text=True).splitlines()
    actual_point = [float(v) for v in lines[0].split()]
    fx = 1280 / (2 * math.tan(math.radians(90) / 2))
    expected_point = [(704 - 640) * 10 / fx, (392 - 360) * 10 / fx, 10]
    assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(actual_point, expected_point))
    actual_roi = [float(v) for v in lines[1].split()]
    pixels = [(x, y) for y in range(2, 6) for x in range(2, 6)]
    roi_expected = [median([(x - 4) * 10 / 4 for x, _ in pixels]),
                    median([(y - 4) * 10 / 4 for _, y in pixels]), 10, 16]
    assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(actual_roi, roi_expected))
    print('PASS: Python/C++ synthetic fixture parity')


if __name__ == '__main__':
    main(sys.argv[1])
