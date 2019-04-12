"""Scene detection algorithms

references:
https://bcastell.com/posts/scene-detection-tutorial-part-1/
"""
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import json


def naive(cap: cv2.VideoCapture, **kwargs) -> []:
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
        threshold = 5
        if means and abs(frame_mean - means[-1]) > threshold:
            frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cuts.append(frame_no)

        means.append(frame_mean)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    return cuts


def naive_double_cap(cap: cv2.VideoCapture, **kwargs) -> []:
    """Compute the time of the cuts in the video,
    :return a list of the frame numbers where cut occurs
    """
    tb = 22
    ts = 2

    means = []
    cuts = []
    Fs = None
    while True:
        (rv, im) = cap.read()  # im is a valid image if and only if rv is true
        if not rv:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        means.append(im.mean())
        if len(means) == 1:
            continue
        delta = abs(means[-2] - means[-1])
        if delta > tb:
            cuts.append(frame_no)
            Fs = None
        elif delta > ts:
            if not Fs:
                Fs = means[-1]
            elif abs(means[-1] - Fs) > tb:
                cuts.append(frame_no)
                Fs = None
        else:
            Fs = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    return cuts


def fade_cut_generator(cap: cv2.VideoCapture, threshold=100):
    means = []
    while True:
        rv, im = cap.read()
        if not rv:
            break
        current_mean = im.mean()

        if means and ((current_mean >= threshold and means[-1] < threshold)
                      or (current_mean < threshold and means[-1] >= threshold)):
            yield int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        means.append(current_mean)


def fade_cuts(cap: cv2.VideoCapture, **kwargs) -> []:
    """ Compute the times of cuts in the video using the tutorial
    https://bcastell.com/posts/scene-detection-tutorial-part-1/
    which is used to solve the problem of fade-cuts getting passed the naive
    algorithm.
    """

    threshold = kwargs.get('threshold', 100)
    # Doing it with a for-loop simply to get prints as it finds them, otherwise,
    # just replace with:
    # return list(fade_cut_generator(cap, threshold)

    cuts = []
    for cut in fade_cut_generator(cap, threshold):
        # print("cut found at {}".format(cut))
        cuts.append(cut)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    return cuts


def hybrid(cap: cv2.VideoCapture, **kwargs) -> []:
    return sorted(naive(cap, **kwargs) + fade_cuts(cap, **kwargs))


def manhattan_distance(l1, l2):
    return sum([abs(l1[i] - l2[i]) for i in range(4)])


def multimean(im):
    w = im.shape[0]
    h = im.shape[1]
    top_left = im[:w // 2, :h // 2, :]
    top_right = im[w // 2:, :h // 2, :]
    bottom_left = im[:w // 2, h // 2:, :]
    bottom_right = im[w // 2:, h // 2:, :]
    return [
        top_left.mean(),
        top_right.mean(),
        bottom_left.mean(),
        bottom_right.mean(),
    ]


def multimean_cuts_generator(cap: cv2.VideoCapture, **kwargs) -> []:
    threshold = kwargs.get('threshold', 30)
    multimeans = []
    while True:
        rv, im = cap.read()
        if not rv:
            break

        current_multimean = multimean(im)

        if multimeans and abs(manhattan_distance(multimeans[-1],
                                                 current_multimean)) > threshold:
            yield int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        multimeans.append(current_multimean)


def multimean_cuts(cap: cv2.VideoCapture, **kwargs) -> []:
    cuts = []
    for cut in multimean_cuts_generator(cap, **kwargs):
        cuts.append(cut)

    return cuts


def edge_detection(cap: cv2.VideoCapture, **kwargs) -> []:
    cuts = []
    i = 0
    D_list = []
    E_list = []
    output_object = []
    while True:
        (rv, im) = cap.read()  # im is a valid image if and only if rv is true
        if not rv:
            break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert from BGR to RGB

        i += 1
        if False and i > 10:
            print("Breaking early for testing")
            break

        E = cv2.Canny(im, 0, 500)
        D = expand_edges(E)

        # np.multiply est le element-wise product, pas le produit matriciel
        # (parce que je l'oubli tout le temps.
        if D_list and E_list:
            SE = np.sum(E)
            rho_in = np.sum(np.multiply(D_list[-1], E)) / SE
            rho_out = np.sum(np.multiply(E_list[-1], D)) / SE
            # print("i = {}, rho_in = {}, rho_out = {}".format(i, rho_in, rho_out))
            output_object.append((i, rho_in, rho_out))

        D_list.append(D)
        E_list.append(E)

    with open('rho_value.json', 'w+') as f:
        f.write(json.dumps(output_object, indent=2))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    return cuts


def edge_detection2(cap: cv2.VideoCapture, **kwargs) -> []:
    cuts = []
    i = 0
    D_list = []
    deltas = []
    last_delta = 0
    while True:
        (rv, im) = cap.read()  # im is a valid image if and only if rv is true
        if not rv:
            break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert from BGR to RGB

        E = cv2.Canny(im, 0, 500)
        D_list.append(255 - expand_edges_fast(E))
        if len(D_list) < 2:
            continue
        delta = np.sum(np.multiply(E, D_list[-2])) / np.sum(D_list[-2]) * 100

        deltas.append(delta)
        match = abs(delta-last_delta) > 0.5
        if match:
            cuts.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        i += 1
        # print("{}: {} ".format(i, delta-last_delta))
        last_delta = delta

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video for further uses
    # plt.plot(deltas), plt.show()
    return cuts


def expand_edges(img: np.array) -> np.array:
    neighbors = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    return ndimage.generic_filter(
        img,
        lambda x: 0 if x.any() else 1,
        footprint=neighbors)


def expand_edges_fast(img: np.array) -> np.array:
    return cv2.GaussianBlur(img, (3, 3), 0)


def edge_detection_cached(cap: cv2.VideoCapture, **kwargs) -> []:
    with open('rho_value.json') as f:
        rho_values = json.loads(f.read())

    threshold = 0.8
    intervals = []
    rho_max_values = [max(d[1], d[2]) for d in rho_values]
    L = len(rho_max_values)
    i = 0
    while i < L:

        rho = rho_max_values[i]
        if rho > threshold:
            start = i
            while i < L and rho_max_values[i] > threshold:
                i += 1

            intervals.append((start, i))

        i += 1

    print(intervals)
    return [int((interval[0] + interval[1])/2.0) for interval in intervals]


def plot_array(arr, title='image') -> None:
    plt.plot(122), plt.imshow(arr, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

