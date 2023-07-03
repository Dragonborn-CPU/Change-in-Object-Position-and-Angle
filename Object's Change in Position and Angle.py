import cv2
import numpy as np
import math


def find_object_change(image_path1, image_path2, pixels_per_mm, center_region=130, num_matches=10):
    # Load the two images
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Resize the images
    resized_image1 = cv2.resize(image1, (640, 480))
    resized_image2 = cv2.resize(image2, (640, 480))

    # edit these values to change keypoints found
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=2.1, nlevels=8, edgeThreshold=10)

    keypoints1, descriptors1 = orb.detectAndCompute(resized_image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(resized_image2, None)

    # Filter keypoints not in the center region, optional can remove/change
    filtered_keypoints1 = [kp for kp in keypoints1 if not (
            resized_image1.shape[1] // 2 - center_region <= kp.pt[0] <= resized_image1.shape[1] // 2 + center_region
            and resized_image1.shape[0] // 2 - center_region <= kp.pt[1] <=
            resized_image1.shape[0] // 2 + center_region)]

    filtered_keypoints2 = [kp for kp in keypoints2 if not (
            resized_image2.shape[1] // 2 - center_region <= kp.pt[0] <= resized_image2.shape[1] // 2 + center_region
            and resized_image2.shape[0] // 2 - center_region <= kp.pt[1] <=
            resized_image2.shape[0] // 2 + center_region)]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    matches = sorted(matches, key=lambda x: x.distance)

    # Get the matched keypoints
    matched_keypoints1 = []
    matched_keypoints2 = []

    for match in matches:
        query_idx = match.queryIdx
        train_idx = match.trainIdx

        if query_idx < len(filtered_keypoints1) and train_idx < len(filtered_keypoints2):
            matched_keypoints1.append(filtered_keypoints1[query_idx].pt)
            matched_keypoints2.append(filtered_keypoints2[train_idx].pt)

    matched_keypoints1 = np.array(matched_keypoints1).reshape(-1, 1, 2)
    matched_keypoints2 = np.array(matched_keypoints2).reshape(-1, 1, 2)

    # Find the similarity transform matrix
    transform, _ = cv2.estimateAffinePartial2D(matched_keypoints1, matched_keypoints2)

    # Extract translation and rotation from the transform matrix NOTE: remove pixels_per_mm and replace with 1 for dx and dy if you are not measuring in inches/mm
    dx = (transform[0, 2] / pixels_per_mm)
    dy = (transform[1, 2] / pixels_per_mm)
    angle = np.arctan2(transform[1, 0], transform[0, 0])

    # Draw the top matches, optional, for displaying image
    result = cv2.drawMatches(resized_image1, filtered_keypoints1, resized_image2, filtered_keypoints2,
                             matches[:num_matches], None, flags=2)

    # Add text showing the change in position and angle on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"dx: {dx:.2f}, dy: {dy:.2f}", (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(result, f"change in angle: {np.degrees(angle):.2f} degrees", (20, 80), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with keypoints
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dx, dy, angle


# Function to calculate the pixels per millimeter and width - requires an object in the image to have a defined and known length
def calculate_pixels_per_mm(image_path, ruler_length_mm):
    # Load the image
    image = cv2.imread(image_path)
    resized_image3 = cv2.resize(image, (640, 480))

    gray = cv2.cvtColor(resized_image3, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 30, 70)  # Adjust the threshold values as needed

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the image
    # image_with_contour = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    ruler_length_pixel = math.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2)

    pixels_per_mm = ruler_length_pixel / ruler_length_mm

    # Show the image with the largest contour
    # cv2.imshow("Image with Largest Contour", image_with_contour)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return pixels_per_mm

# images used, use correct file path
image_path1 = 'C:/Users/ethan/Desktop/Len16/Backup/220921_034158_0000000001_&Cam1Img.bmp'
image_path2 = 'C:/Users/ethan/Desktop/Len25/220921_032307_0000000004_&Cam1Img.bmp'
ruler_length_mm = 378  # 538   # Length of the ruler in inches

#image used with object with known length for length calculations - must be one of the two images being compared
ruler_image = 'C:/Users/ethan/Desktop/Len16/Backup/220921_034158_0000000001_&Cam1Img.bmp'
pixels_per_mm = calculate_pixels_per_mm(ruler_image, ruler_length_mm)

#print information
dx, dy, angle = find_object_change(image_path1, image_path2, pixels_per_mm)
print(f"Change in position (mm): dx = {dx:.2f}, dy = {dy:.2f}")
print(f"Change in \u03B8: {angle:.2f} degrees")
