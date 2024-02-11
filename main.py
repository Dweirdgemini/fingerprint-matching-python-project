import cv2 as cv
import os

# fingerprints file directory
filepath = "fingerprints/result"

# read single sample of fingerprint file
sample = cv.imread(f"{filepath}/1_5.bmp")

# list of all fingerprints filepath
fingerprints = [file for file in os.listdir(filepath)][1:]

# create instance of Scale Invariant Feature Transform (SIFT)
sift = cv.SIFT_create()

# Detect Keypoints in fingerprint and its corresponding descriptor
keypoints_1, descriptor_1 = sift.detectAndCompute(sample, None)

scores = {}
best_score = [0]
# iterate over list of fingerprint filepath list
for fingerprint in fingerprints:

    file = f"{filepath}/{fingerprint}"
    filename = fingerprint
    fingerprint = cv.imread(file)
    keypoints_2, descriptor_2 = sift.detectAndCompute(fingerprint, None)

    # Perform Fast Local Approximate Nearest Neighbour (FLANN) calculation between fingerprints
    matches = cv.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(descriptor_1, descriptor_2, k=2)

    # create list to store matched points
    match_points = []

    # Compare distance between the matched points
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)
    
    
    keypoints = 0

    # assign keypoints to length of the larger one among the two sets of keypoints
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    # calculate match score and store in scores dictionary
    if len(match_points) / keypoints  * 100 > best_score[-1]:
        scores.clear()
        best_score.append(len(match_points) / keypoints  * 100)
        scores[filename] = best_score[-1]
        image = fingerprint
        kp1 = keypoints_1
        kp2 = keypoints_2
        mp = match_points

# Print fingerprint match
match = list(scores.keys())[0]
print(f"Best Match: {match}")
print(f"Score: {scores[match]}")

result = cv.drawMatches(sample, kp1, image, kp2, mp, None)
#result = cv.resize(result, None, fx=4, fy=4)

cv.imshow("Result", result)
cv.waitKey(0)
cv.destroyAllWindows()