# Change-in-Object-Position-and-Angle
Find a change in a object's position and angle between two images. 

Uses similarity matrix instead of homography matrix in case object is scaled or sheared in one of the images (and not just rotated or translated).

Two different functions, calculate_pixels_per_mm is an optional function for calculating an object's change in position in exact units

Main function is find_object_change which has multiple different vairables and finds/matches keypoints to determine the change in angle and position. Multiple different variable's can be changed, including center_region to exclufe keypoints from the center and num_macthes for number of keypoints used. 
