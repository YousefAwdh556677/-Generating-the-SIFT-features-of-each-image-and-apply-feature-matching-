import cv2
import matplotlib.pyplot as plt

# Load the images
petra1 = cv2.imread('petra1.jpg')
petra2 = cv2.imread('petra2.jpg')
petra3 = cv2.imread('petra3.jpg')
petra4 = cv2.imread('petra4.jpg')

gray1 = cv2.cvtColor(petra1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(petra2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(petra3, cv2.COLOR_BGR2GRAY)  # Corrected petra2 to petra3 here
gray4 = cv2.cvtColor(petra4, cv2.COLOR_BGR2GRAY)

# Create the SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Detect SIFT features for each image
kp1, des1 = sift.detectAndCompute(petra1, None)
kp2, des2 = sift.detectAndCompute(petra2, None)
kp3, des3 = sift.detectAndCompute(petra3, None)
kp4, des4 = sift.detectAndCompute(petra4, None)

# Display SIFT features on images
petra1_sift = cv2.drawKeypoints(gray1, kp1, None)
petra2_sift = cv2.drawKeypoints(gray2, kp2, None)
petra3_sift = cv2.drawKeypoints(gray3, kp3, None)
petra4_sift = cv2.drawKeypoints(gray4, kp4, None)

cv2.imshow("I'm Sift One", petra1_sift)
cv2.imshow("I'm Sift Two", petra2_sift)
cv2.imshow("I'm Sift Three", petra3_sift)
cv2.imshow("I'm Sift Four", petra4_sift)

# Feature matching using Brute Force Matcher
bf = cv2.BFMatcher()
matches_1_2 = bf.knnMatch(des1, des2, k=2)
matches_1_3 = bf.knnMatch(des1, des3, k=2)
matches_1_4 = bf.knnMatch(des1, des4, k=2)

# Apply ratio test and filter good matches
good_matches_1_2 = [m for m, n in matches_1_2 if m.distance < 0.75 * n.distance]
good_matches_1_3 = [m for m, n in matches_1_3 if m.distance < 0.75 * n.distance]
good_matches_1_4 = [m for m, n in matches_1_4 if m.distance < 0.75 * n.distance]

# Draw matches
img_matches_1_2 = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches_1_2, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_matches_1_3 = cv2.drawMatches(gray1, kp1, gray3, kp3, good_matches_1_3, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_matches_1_4 = cv2.drawMatches(gray1, kp1, gray4, kp4, good_matches_1_4, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Plot matches
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(img_matches_1_2)
plt.title('Matches between img1 and img2')
plt.axis('off')

plt.subplot(132)
plt.imshow(img_matches_1_3)
plt.title('Matches between img1 and img3')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_matches_1_4)
plt.title('Matches between img1 and img4')
plt.axis('off')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
