#!/usr/bin/env python3
"""Small, self-contained smoke test for the 66+1 feature interface.

Creates a short synthetic screen-content video. No benchmark scores, models,
external downloads, or GPU are involved. This is an interface test only.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import cv2

from litescvqa_features import extract_video_features, feature_names

def main():
    with TemporaryDirectory() as d:
        path = str(Path(d) / "synthetic.avi")
        w, h = 320, 240
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 12, (w, h))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create synthetic MJPG video")
        for i in range(24):
            img = np.full((h, w, 3), 235, dtype=np.uint8)
            cv2.putText(img, f"Slide {i//8+1}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
            cv2.rectangle(img, (30 + i, 130), (120 + i, 165), (30, 80, 190), 2)
            writer.write(img)
        writer.release()
        features, info = extract_video_features(path)
    assert len(feature_names()) == 66, len(feature_names())
    assert len(features) == 66, len(features)
    assert np.isfinite(features).all()
    assert info["frames"] == 24, info
    assert 0 < info["key_ratio"] <= 1, info
    print("PASS: 66 extracted values + 1 separately exported key_ratio = 67 model inputs")
    print(f"frames={info['frames']} key_frames={info['key_frames']} ratio={info['key_ratio']:.4f}")

if __name__ == "__main__":
    main()
