import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to feather the transition between two images using gradient mask
def create_blend_mask(width_left, panorama_width, height):
    mask = np.zeros((height, panorama_width), dtype=np.float32)
    blend_width = width_left // 5  # Define a blending width (adjustable based on the overlap region)

    mask[:, :width_left - blend_width] = 1.0  # Left image fully included
    for i in range(blend_width):
        mask[:, width_left - blend_width + i] = 1.0 - (i / blend_width)  # Gradient for blending

    return cv2.merge([mask, mask, mask])  # Return a 3-channel mask for blending

# Load the two images
left_image = cv2.imread(r'C:\Users\dirky\Desktop\diner_left.JPG')
right_image = cv2.imread(r'C:\Users\dirky\Desktop\diner_right.JPG')

# (b) Find the SIFT-key points and descriptors for both images
# Convert images to grayscale
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

# (c) Match the correspondence points using Brute-Force matcher with cross-checking
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors_left, descriptors_right)

# Sort matches based on their distances (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
matched_image = cv2.drawMatches(left_image, keypoints_left, right_image, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matched keypoints
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Matched Keypoints between Left and Right Images')
plt.show()

# (d) Run RANSAC to estimate homography
# Extract location of keypoints
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate homography using RANSAC
H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC)

# Warp the right image to align with the left image
height_left, width_left = left_image.shape[:2]
panorama_width = width_left + right_image.shape[1]
panorama_height = max(height_left, right_image.shape[0])

# Warp the right image using the homography matrix
warped_right = cv2.warpPerspective(right_image, H, (panorama_width, panorama_height))

# Place the left image in the panorama
panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
panorama[0:height_left, 0:width_left] = left_image

# Create a gradient mask for feathering
blend_mask = create_blend_mask(width_left, panorama_width, panorama_height)

# Perform the blending operation using the mask
blended_panorama = panorama * blend_mask + warped_right * (1 - blend_mask)

# Convert the result to uint8 type for display
blended_panorama = blended_panorama.astype(np.uint8)

# Show the feather-blended panorama
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(blended_panorama, cv2.COLOR_BGR2RGB))
plt.title('Feather Blended Stitched Panorama')
plt.show()