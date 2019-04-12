""" Scene detection on video sequence using different algorithms


"""
import sys
import cv2
from pprint import pprint

import detectionalgo as algo
import util


class Video:
    def __init__(self, filename):
        self.cap = cv2.VideoCapture()
        self.cap.open(filename)
        if not self.cap.isOpened():
            raise ValueError(filename)
        self.detection_algo = getattr(algo, 'naive')
        # print(self.get_dimensions())
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def __del__(self):
        self.cap.release()

    def set_algo(self, algo_name):
        try:
            self.detection_algo = getattr(algo, algo_name)
        except:
            sys.exit('algorithm not found')

    def get_cuts(self, **kwargs) -> []:
        """Compute the time of the cuts in the video,
        :return a list of the frame numbers where cut occurs
        """
        cuts = self.detection_algo(self.cap, **kwargs)
        return cuts


def run_algo(video, algo, threshold) -> None:
    video.set_algo(algo)
    cuts = video.get_cuts(threshold=threshold)
    report(algo, cuts)


def report(algo, cuts) -> None:
    print("*" * 40)
    print("Using algo: '{}'".format(algo))
    # print(cuts)
    print("Cross checking with ground truth:")
    util.verify_result(cuts, 10)
    false_pos = util.count_false_positives(cuts)
    print("false positives : {} ".format(len(false_pos)) + str(false_pos))


def main():
    if len(sys.argv) < 2:
        print("Error - file name must be specified as first argument.")
        return

    filename = sys.argv[1]
    try:
        video = Video(filename)
    except ValueError:
        print("Could not open {}", filename)
        return

    # run_algo(video, 'naive', None)
    # run_algo(video, 'fade_cuts', 100)
    # run_algo(video, 'edge_detection_cached', None)
    run_algo(video, 'edge_detection', None)


if __name__ == "__main__":
    main()
