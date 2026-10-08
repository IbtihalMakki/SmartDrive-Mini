"""Decode raw CARLA depth and pair camera measurements by simulation frame."""

import threading
import time

import numpy as np


def decode_depth(image):
    """Decode raw BGRA (not color-converted visualization) into meters."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    ).astype(np.float32)
    return (bgra[:, :, 2] + 256.0 * bgra[:, :, 1]
            + 65536.0 * bgra[:, :, 0]) * (1000.0 / 16777215.0)


class RGBDFrameBuffer:
    """Bounded, thread-safe pairing; each frame can be consumed only once."""

    def __init__(self, capacity=8, max_age=2.0, validation=False):
        self.capacity = capacity
        self.max_age = max_age
        self.frames = {"rgb": {}, "depth": {}}
        self.validation = validation
        if validation:
            self.frames["reference"] = {}
        self.lock = threading.Lock()
        self.last_frame = -1

    def add(self, kind, image):
        with self.lock:
            if image.frame <= self.last_frame:
                return
            queue = self.frames[kind]
            queue[image.frame] = (image, time.monotonic())
            while len(queue) > self.capacity:
                del queue[min(queue)]

    def pop(self):
        with self.lock:
            now = time.monotonic()
            for queue in self.frames.values():
                for frame in list(queue):
                    if now - queue[frame][1] > self.max_age:
                        del queue[frame]
            common = self.frames["rgb"].keys() & self.frames["depth"].keys()
            if self.validation:
                common &= self.frames["reference"].keys()
            if not common:
                return None
            frame = max(common)
            rgb = self.frames["rgb"][frame][0]
            depth = self.frames["depth"][frame][0]
            reference = self.frames["reference"][frame][0] if self.validation else None
            self.last_frame = frame
            for queue in self.frames.values():
                for old_frame in list(queue):
                    if old_frame <= frame:
                        del queue[old_frame]
        if (rgb.width, rgb.height) != (depth.width, depth.height):
            raise ValueError("RGB and depth image dimensions do not match")
        if abs(rgb.timestamp - depth.timestamp) > 1e-6:
            raise ValueError("RGB and depth timestamps do not match")
        bgr = np.frombuffer(rgb.raw_data, dtype=np.uint8).reshape(
            rgb.height, rgb.width, 4
        )[:, :, :3].copy()
        result = (frame, rgb.timestamp, bgr, decode_depth(depth))
        if self.validation:
            if abs(rgb.timestamp - reference.timestamp) > 1e-6:
                raise ValueError("Reference timestamp does not match RGB-D")
            # Camera and LiDAR share the same rigid mounting transform.
            if not np.allclose(rgb.transform.get_matrix(), reference.transform.get_matrix(), atol=1e-4):
                raise ValueError("Reference sensor pose does not match RGB camera")
            return result + (reference,)
        return result
