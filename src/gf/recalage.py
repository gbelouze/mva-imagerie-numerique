import cv2  # type: ignore
import numpy as np


def make_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float64:
        return (255 * image).astype(np.uint8)
    elif image.dtype == np.int64:
        return image.astype(np.uint8)
    assert image.dtype == np.uint8
    return image


def findHomography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Computes homography transforming [source] into [target]
    Code taken from : https://www.geeksforgeeks.org/image-registration-using-opencv-python/
    """
    assert source.shape == target.shape
    assert 2 <= source.ndim <= 3

    source_gray = cv2.cvtColor(make_uint8(source), cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(make_uint8(target), cv2.COLOR_BGR2GRAY)

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(source_gray, None)
    kp2, d2 = orb_detector.detectAndCompute(target_gray, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[: int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography


def registration(*imgs: np.ndarray) -> list[np.ndarray]:
    """Registration for an arbitrary number n of images.
    Images are taken in order and successively transformed in the reference frame of the following one
    (hence order of arguments matter !). Warning : errors will propagate if the number of images is too large.
    Returns n images, the inputs expressed in the reference frame of the last image.
    """
    shape = imgs[0].shape
    height, width = shape[:2]
    assert all(img.shape == shape for img in imgs)

    onestep_homographies = []
    for source, target in zip(imgs[:-1], imgs[1:]):
        onestep_homographies.append(findHomography(source, target))

    homographies = [np.identity(3)]
    for onestep_homography in onestep_homographies[::-1]:
        homographies.append(onestep_homography @ homographies[-1])
    homographies = homographies[::-1]

    return [
        cv2.warpPerspective(img, homography, (width, height))
        for img, homography in zip(imgs, homographies)
    ]


def misalignment(warped: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Populates an image [out] such that [out] is gray where [warped] and [target]
    match, and colourful elsewhere."""
    assert warped.shape == target.shape
    assert 2 <= warped.ndim <= 3
    coloured = warped.ndim == 3

    out = np.empty(warped.shape) if coloured else np.empty((*warped.shape, 3))
    out[:, :, 0] = warped.mean(axis=2) if coloured else warped
    out[:, :, 1] = target.mean(axis=2) if coloured else target
    out[:, :, 2] = (out[:, :, 0] + out[:, :, 1]) / 2

    return out
