import os
import cv2
import time
import shutil


def video2frames(path, skip=0):
    result_folder = "./frames"

    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)

    vidcap = cv2.VideoCapture(path)
    success = True

    save_count = 0
    frame_count = 0

    while success:
        success, image = vidcap.read()
        if frame_count % skip == 0:
            cv2.imwrite(f"frames/frame{save_count}.jpg", image)
            print(f"Saved Frame: {save_count}")
            save_count += 1

        frame_count += 1


if __name__ == "__main__":
    video2frames("./video/All Lights On 2.MOV", skip=10)
