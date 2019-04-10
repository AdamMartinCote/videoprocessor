"""Scene detection algorithms

references:
https://bcastell.com/posts/scene-detection-tutorial-part-1/
"""
import cv2


def naive(cap) -> []:
    """Compute the time of the cuts in the video,
    :return a list of the frame numbers where cut occurs
    """
    means = []
    cuts = []
    while True:
        (rv, im) = cap.read()  # im is a valid image if and only if rv is true
        if not rv:
            break
        frame_mean = im.mean()
        threshold = 20
        if means and abs(frame_mean - means[-1]) > threshold:
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cuts.append(frame_no)

        means.append(frame_mean)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    return cuts