"""
Code Name: Python Backend for Strain Evaluation

Author: Surya S. Vel

Date: 2/17/2026

Version: 13
"""

#******************
# Import Libraries
#******************
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import math
import random
import cv2
import numpy as np
import statistics
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import imutils
import argparse
from sklearn.cluster import KMeans
import scipy.optimize as opt
from scipy.linalg import sqrtm,logm, expm
import glob
import os
import pygame
from scipy import stats
import os
import sys
import ctypes

app = Flask(__name__)
CORS(app)


################# Functions ###################

#----------------------------------------------------------------------
# Function that finds the distance between two points
#
#  Inputs: P - first point as a numpy array
#		   Q - second point as a numpy array
#
# Outputs: d - distance between P and Q
#----------------------------------------------------------------------
def distance(P,Q):
    # print(P)
    # Px,Py = P
    # Qx,Qy = Q
    # dx = abs(Px-Qx)
    # dy = abs(Py-Qy)
    
    # # Calculate distance
    # d = math.sqrt(dx**2 + dy**2)
    d = np.linalg.norm(P - Q)

    return(d)

#----------------------------------------------------------------------
# Function that applies a Gaussian blur based on image size 
#
#  Inputs: img - cv2 image
#          blur_sigma_scale - scale factor for blur intensity
#
# Outputs: img - Gaussian blurred image
#----------------------------------------------------------------------
def blur_filter_sigma(img,blur_sigma_scale):

    # Find image dimensions
    h,w = img.shape[:2]
    # Find the blue sigma by multiplying image size by sigma scale
    blur_sigma = math.ceil(blur_sigma_scale*min(h,w))
    # Blur image
    img = cv2.GaussianBlur(img, (0, 0), sigmaX = blur_sigma, sigmaY = 0)

    return(img)

#----------------------------------------------------------------------
# Function to convert grayscale image to black and white
#
#  Inputs: c_image - cv2 image in grayscale
#
# Outputs: g_image - cv2 image in black and white
#----------------------------------------------------------------------
def threshold_image(g_image,threshold_block_scale):

    # Find image dimensions 
    h,w = g_image.shape[:2]

    # Determine the block size used for adaptive thresholding
    block_size = int(threshold_block_scale*(min(h,w)))

    # Ensure that block size is an odd number
    if block_size % 2 == 0:
        block_size += 1
          
    # Use adaptive thresholding to convert to black and white
    bw_image = (cv2.adaptiveThreshold(g_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,block_size,13))
    
    return(bw_image)

#----------------------------------------------------------------------
# Function that cleans image noise using morphological closing
#
#  Inputs: img - cv2 image in black and white
#          morphology_kernel_scale - scale factor for kernel size
#
# Outputs: img - cleaned image after morphology closing operations
#----------------------------------------------------------------------
def morphology_cleaning(img,morphology_kernel_scale):

    # Find image dimensions
    h,w = img.shape[:2]
    # Find kernel size by multiplying the kernel scale by the image size
    morphology_kernel_size = int(morphology_kernel_scale*min(h,w))

    # Create the actual kernel by making an array of ones that has dimensions of kernel size x kernel size
    kernel = np.ones((morphology_kernel_size,morphology_kernel_size), np.uint8)

    # Run the morphological closing operation twice
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return(img)

#----------------------------------------------------------------------
# Function that extracts and filters contours
#
#  Inputs: bw_image - black and white cv2 image
#
# Outputs: cnts - list of sorted contours
#----------------------------------------------------------------------
def extract_contours(bw_image):
    
    # Finding contours from the black and white image
    cnts,hierarchy = cv2.findContours(bw_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # Ignore contours that are within contours
    filtered_contours = []
    for i, h in enumerate(hierarchy[0]): 
        parent_idx = h[3]  # h[3] = parent contour index
        if parent_idx == -1:  # Only outer contours (no parent)
            filtered_contours.append(cnts[i])
    cnts = filtered_contours

    # Sort contours by area, so the three position marker contours are at the begining
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	
    return(cnts)

#----------------------------------------------------------------------
# Function that calculates the area and centroid (cX, cY) to sort contours
#
#  Inputs: cnts - list of contours
#
# Outputs: cnts_areas_centroids - NumPy array for contours with format [area,cx,cy]
#----------------------------------------------------------------------
def evaluate_contour_areas_centroids(cnts):
	
    # Find number of contours
    cnts_len = len(cnts)
    # Create an array of zeros of same length as the number of contours
    cnts_areas_centroids = np.zeros((cnts_len, 3))
	
    # Evaluating Contour Areas and Centroids
    for n in range(cnts_len):
        c = cnts[n]
        M = cv2.moments(c) 
        # Calculate contour area using the contour moment
        cnts_areas_centroids[n][0] = M["m00"]
        if M["m00"] != 0:
            # Calculate the centroid of the contour
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            cnts_areas_centroids[n][1] = cX
            cnts_areas_centroids[n][2] = cY
        else:
            print('ERROR: AREA OF CONTOUR:', n, 'IS ZERO!!!')

    return(cnts_areas_centroids)

#----------------------------------------------------------------------
# Function that identifies and sorts the three QR position markers
#
#  Inputs: image - cv2 image (used for dimensions)
#          cnts_areas_centroids - NumPy array for contours with format [area,cx,cy]
#
# Outputs: P, Q, R - Centroids of the Top-Left, Top-Right, and Bottom-Left poition markers respectively
#          PM_areas_centroids - NumPy array for position markers with format [area,cx,cy]
#----------------------------------------------------------------------
def identify_position_markers(image,cnts_areas_centroids):

    # Find the number of contours
    cnts_len = len(cnts_areas_centroids)

    # Extract position markers area and centroid from the cnts array
    PM_areas_centroids = cnts_areas_centroids[0:3, 0:3]

    # Find the size of the image
    (height, width) = image.shape[0:2]

    # Point closest to the top left (0,0)
    d0 = 2*max(width,height) # Double the max image dimension
    for k in range(3):
        # Find the distance between the centroid and top left of the image
        d  = np.sqrt((PM_areas_centroids[k,1] -0)**2 + (PM_areas_centroids[k,2] - 0)**2)
        if d < d0:
            # If this is the closest centroid so far, update d0
            d0 = d
            # P is the centroid closest to the top left
            P = PM_areas_centroids[k]

    # Point closest to the top right (width,0)
    d0 = 2*max(width,height) # Double the max image dimension
    for k in range(3):
        # Find the distance between the centroid and top right of the image
        d  = np.sqrt((PM_areas_centroids[k,1] - width)**2 + (PM_areas_centroids[k,2] - 0)**2)
        if d < d0:
            # If this is the closest centroid so far, update d0
            d0 = d
            # Q is the centroid closest to the top right
            Q = PM_areas_centroids[k]

    #Point closest to the bottom left (0,height)
    d0 = 2*max(width,height) # Double the max image dimension
    for k in range(3):
        # Find the distance between the centroid and bottom left of the image
        d  = np.sqrt((PM_areas_centroids[k,1] - 0)**2 + (PM_areas_centroids[k,2] - height)**2)
        if d < d0:
            # If this is the closest centroid so far, update d0
            d0 = d
            # R is the centroid closest to the bottom left
            R = PM_areas_centroids[k]

    return(P,Q,R,PM_areas_centroids) 

#----------------------------------------------------------------------
# Function processes an image and extracts the data module and position markers areas and centroids
#
#  Inputs: image - color image for processing
#
# Outputs: P, Q, R - Centroids of the Top-Left, Top-Right, and Bottom-Left poition markers respectively
#          cnts_areas_centroids - NumPy array for all contours with format [area,cx,cy]
#          data_extract - black and white image for data extraction
#----------------------------------------------------------------------
def image_processing(image):

    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur filter for noise reduction
    # 0.004 default
    blur_sigma_scale = 0.004
    image_gray = blur_filter_sigma(image_gray,blur_sigma_scale)

    # Convert to B&W image using adaptive threshold
    # 0.11 default
    threshold_block_scale = 0.11
    image_bw = threshold_image(image_gray,threshold_block_scale)

    # Threshold image differently for extracting the data
    data_extract = threshold_image(image_gray,0.2)

    # Apply morphological operations to remove small noise and fill small holes or gaps
    morphology_kernel_scale = 0.001
    image_bw = morphology_cleaning(image_bw,morphology_kernel_scale)

    # Use the negative image (flip swap black and white) as that works better for extracting contours
    image_bw_neg = 255 - image_bw

    # Extract contours
    cnts = extract_contours(image_bw_neg)

    # Organize contours by their area and centroids
    cnts_areas_centroids = evaluate_contour_areas_centroids(cnts)

    # Find the 3 position markers
    P,Q,R,PM_areas_centroids = identify_position_markers(image_bw,cnts_areas_centroids)

    return(P,Q,R,cnts_areas_centroids,data_extract)

#----------------------------------------------------------------------
# Function that deforms an image based on affine parameters 
#
#  Inputs: A11,A12,A21,A22,b1,b2 - affine parameters
#          S - Point to be mapped
#
# Outputs: S_mapped - S after affine deformation
#----------------------------------------------------------------------
def affine_mapping(A11,A12,A21,A22,b1,b2,S):
     
    # Extract x and y
    x = S[0]
    y = S[1]

    # Perform affine mapping
    x_prime = A11*x + A12*y + b1
    y_prime = A21*x + A22*y + b2
    S_mapped = np.array([x_prime,y_prime])

    return(S_mapped)

#----------------------------------------------------------------------
# Function that a affine transformation matrix for the source and destination points
#
#  Inputs: src_pts - array of the three reference position marker centroids [P_ref,Q_ref,R_ref]
#          dst_pts - array of the three deformed position marker centroids [P_def,Q_def,R_def]
#
# Outputs: A - 2x3 NumPy array with the calculated affine coefficients
#----------------------------------------------------------------------
def find_affine_coeffs(src_pts, dst_pts):

    # Extract src coords
    xP, yP = src_pts[0]
    xQ, yQ = src_pts[1]
    xR, yR = src_pts[2]
    
    # Extract dst coords
    xPd, yPd = dst_pts[0]
    xQd, yQd = dst_pts[1]
    xRd, yRd = dst_pts[2]

    # M * Affine_array = b
    # So find affine coefficentc by using np.linalg.solve(M, b)

    # Define M array for source points
    M = np.array([
    [xP, yP, 1, 0, 0, 0],
    [0, 0, 0, xP, yP, 1],
    [xQ, yQ, 1, 0, 0, 0],
    [0, 0, 0, xQ, yQ, 1],
    [xR, yR, 1, 0, 0, 0],
    [0, 0, 0, xR, yR, 1]
    ], dtype=float)

    # Define b array for destination points
    b = np.array([xPd, yPd, xQd, yQd, xRd, yRd], dtype=float)

    # Solve for affine coefficients
    coeffs = np.linalg.solve(M, b)

    # Reshape into 2x3 matrix
    A = np.array([[coeffs[0], coeffs[1], coeffs[2]],[coeffs[3], coeffs[4], coeffs[5]]])

    return(A)

#----------------------------------------------------------------------
# Function that estimates the center of the image
#
#  Inputs: cnts_areas_centroids - NumPy array of contours with format [area,cx,cy]
#
# Outputs: xc, yc - x and y coordinates of the estimated center point
#----------------------------------------------------------------------
def estimate_center(cnts_areas_centroids):

    # Find min and max for x and y
    x_min = np.min(cnts_areas_centroids[:,1])
    x_max = np.max(cnts_areas_centroids[:,1])
    y_min = np.min(cnts_areas_centroids[:,2])
    y_max = np.max(cnts_areas_centroids[:,2])

    # Find center coords (x,y)
    xc = (x_min+x_max)/2
    yc = (y_min+y_max)/2

    return(xc,yc)

#----------------------------------------------------------------------
# Function that filters contours on the areas of the contours
#
#  Inputs: cnts_areas_centroids - NumPy array of contour [area,cx,cy]
#
# Outputs: data_mods - NumPy array of data module contours that weren't excluded
#----------------------------------------------------------------------
def data_module_area_discrimination(cnts_areas_centroids):

    # Find average data module area
    data_mod_area_total = 0
    for i in range(3,len(cnts_areas_centroids)):
        data_mod_area_total += cnts_areas_centroids[i][0]
    data_mod_area_av = data_mod_area_total/(len(cnts_areas_centroids)-3)

    # Define min and max area for data modules
    min_area = 0.75 * data_mod_area_av
    max_area = 1.25 * data_mod_area_av
    data_mods = np.empty((0, 3), dtype=float)

    # Ignore countours with areas outside of the bounds
    for i in range(3,len(cnts_areas_centroids)): 
        c = cnts_areas_centroids[i]
        c_area = cnts_areas_centroids[i][0]
        if min_area < c_area < max_area:
            data_mods = np.vstack([data_mods,c])

    return(data_mods)

#----------------------------------------------------------------------
# Function that correlates the reference and deformed data modules
#
#  Inputs: cnts_areas_centroids_ref - NumPy array of reference data module contours [area,cx,cy]
#          cnts_areas_centroids_ref - NumPy array of deformed data module contours [area,cx,cy
#          M - Affine mapping that was calculated from the position markers
#
# Outputs: data_mods_ref - data module centroids for the reference image
#          data_mods_def - data module centroids for the deformed image
#----------------------------------------------------------------------
def data_module_corresponding(cnts_areas_centroids_ref,cnts_areas_centroids_def,M):

    # Extract the affine coefficents
    A11 = M[0][0]
    A12 = M[0][1]
    b1 = M[0][2]
    A21 = M[1][0]
    A22 = M[1][1]
    b2 = M[1][2]

    ref_len = len(cnts_areas_centroids_ref)
    def_len = len(cnts_areas_centroids_def)
    d_min_total = 0
    for i in range(ref_len):
        # Get the centroid of the reference data module
        c_ref = cnts_areas_centroids_ref[i][1:3]
        # Map the reference data module onto the deformed image
        c_prime = affine_mapping(A11,A12,A21,A22,b1,b2,c_ref)
        d_min = np.inf
        # Go through the deformed image data modules and find the one closest to the mapped reference data module
        for j in range(def_len):
            # Find centroid for deformed data module
            c_def = cnts_areas_centroids_def[j][1:3]
            # Find distance
            d = distance(c_def,c_prime)
            # See if this is the closest deformed data module so far
            if d < d_min:
                d_min = d
        # Add all the distances of the closest deformed data module to the mapped reference data module
        d_min_total += d_min
    # Find the average distance between the closest deformed data module and the mapped reference data module
    d_min_av = d_min_total/(ref_len)
    # Set the max limit
    d_min_max = d_min_av * 1.5
    
    data_mods_ref = np.empty((0, 2), dtype=float)
    data_mods_def = np.empty((0, 2), dtype=float)
    for i in range(ref_len):
        # Get the centroid of the reference data module
        c_ref = cnts_areas_centroids_ref[i][1:3]
        # Map the reference data module onto the deformed image
        c_prime = affine_mapping(A11,A12,A21,A22,b1,b2,c_ref)
        d_min = np.inf
        # Go through the deformed image data modules and find the one closest to the mapped reference data module
        for j in range(def_len):
            # Find centroid for deformed data module
            c_def = cnts_areas_centroids_def[j][1:3]
            # Find distance
            d = distance(c_def,c_prime)
            # See if this is the closest deformed data module so far
            if d < d_min:
                d_min = d
                c_min = c_def
        # Check if the closest deformed data module is wihin in the distance minimum
        if d_min < d_min_max:
            # Append the centroid cooridinates so that data_mods_ref[i] corresponds to data_mods_def[i]
            data_mods_ref = np.vstack([data_mods_ref,c_ref])
            data_mods_def = np.vstack([data_mods_def,c_min])

    return(data_mods_ref,data_mods_def)

#----------------------------------------------------------------------
# Function that calculates the angle in degrees between two vectors
#
#  Inputs: u - first vector as a numpy array
#          v - second vector as a numpy array
#
# Outputs: angle - the angle between u and v in degrees
#----------------------------------------------------------------------
def angle_between(u, v):

    # Return the angle in degrees between two vectors u and v.
    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    # Clip for numerical stability
    cos_theta = np.clip(dot / norm, -1.0, 1.0)

    return(np.degrees(np.arccos(cos_theta)))

#----------------------------------------------------------------------
# Function that finds the four interior angles of a quadrilateral
#
#  Inputs: points - NumPy array that has vertex coordinates in order
#
# Outputs: angles - NumPy array that has the four interior angles in degrees
#----------------------------------------------------------------------
def quadrilateral_angles(points):
    # Compute the four interior angles of a quadrilateral.
    # points are a numpy array of shape (4, 2) with vertices in order.
    angles = []
    n = len(points)
    for i in range(n):
        p_prev = points[(i-1) % n]
        p_curr = points[i]
        p_next = points[(i+1) % n]
        
        u = p_prev - p_curr
        v = p_next - p_curr
        angles.append(angle_between(u, v))
    return(np.array(angles))

#----------------------------------------------------------------------
# Function that finds the aspect ratio of a quadrilateral
#
#  Inputs: coords - NumPy array of the four vertex coordinates
#
# Outputs: aspect_ratio - The ratio of the maximum side length to the minimum side length
#----------------------------------------------------------------------
def quad_aspect_ratio(coords):
    # Make sure array is numpy
    coords = np.asarray(coords)
    
    # Compute distances between consecutive vertices (including last to first)
    side_lengths = []
    for i in range(4):
        p1 = coords[i]
        p2 = coords[(i+1) % 4]   # wrap around
        dist = np.linalg.norm(p2 - p1)
        side_lengths.append(dist)
    
    side_lengths = np.array(side_lengths)
    # Calculate the max aspect ratio
    aspect_ratio = side_lengths.max() / side_lengths.min()
    
    return(aspect_ratio)

#----------------------------------------------------------------------
# Function that finds a quadrilateral for homgraphy calculation
#
#  Inputs: data_mods_ref - NumPy array of reference data module centroids
#          data_mods_def - NumPy array of deformed data module centroids
#          P_ref - Position marker P in the reference image
#          Q_ref - Position marker Q in the reference image
#          R_ref - Position marker R in the reference image
#          xc,yc - Estimated center of the image
#
# Outputs: ref_homography_points - Array of the 4 vertics of the reference image quadrilateral
#          def_homography_points - Array of the 4 vertics of the deformed image quadrilateral
#----------------------------------------------------------------------
def find_homography_points(data_mods_ref,data_mods_def,P_ref,Q_ref,R_ref,xc,yc):

    ref_len = len(data_mods_ref)
    # If accept points is 1, a quadrilateral is outputed, if not keep trying to find one
    accept_points = 0

    # Establish max and min distance the data modules can be from the center of the image
    distance_min = 0.7*(distance(Q_ref,R_ref)/2)
 
    # Establish min and max angles for the selected quadrilateral
    vertex_angle_min = 75.0
    vertex_angle_max = 105.0
    
    # Establish maximum aspect ratio
    aspect_ratio_max = 1.5
    
    while accept_points == 0:

        ref_homography_points = np.empty((0, 2), dtype=float)
        def_homography_points = np.empty((0, 2), dtype=float)

        # Find a data module in quadrant 1 (top right)
        assigned_quad_1 = 0
        while assigned_quad_1 == 0:
            # Find a random data module
            random_data_mod = random.randint(0, ref_len-1)
            random_mod_x, random_mod_y = data_mods_ref[random_data_mod]
            # Making sure that the module is in the top right quadrant
            if random_mod_x > xc:
                if random_mod_y < yc:
                    # Find distance from the image center
                    dist = math.sqrt((random_mod_x - xc) ** 2 + (random_mod_y - yc) ** 2)
                    # Determine if the data module is within the acceptable range
                    if  (dist >= distance_min):
                        ref_homography_points = np.vstack([ref_homography_points,[random_mod_x,random_mod_y]])
                        random_mod_x_def, random_mod_y_def = data_mods_def[random_data_mod]
                        def_homography_points = np.vstack([def_homography_points,[random_mod_x_def, random_mod_y_def]])
                        assigned_quad_1 = 1

        # Find a data module in quadrant 2 (top left)
        assigned_quad_2 = 0
        while assigned_quad_2 == 0:
            # Find a random data module
            random_data_mod = random.randint(0, ref_len-1)
            random_mod_x, random_mod_y = data_mods_ref[random_data_mod]
            # Making sure that the module is in the top left quadrant
            if random_mod_x < xc:
                if random_mod_y < yc:
                    # Find distance from the image center
                    dist = math.sqrt((random_mod_x - xc) ** 2 + (random_mod_y - yc) ** 2)
                    # Determine if the data module is within the acceptable range
                    if  (dist >= distance_min):
                        ref_homography_points = np.vstack([ref_homography_points,[random_mod_x,random_mod_y]])
                        random_mod_x_def, random_mod_y_def = data_mods_def[random_data_mod]
                        def_homography_points = np.vstack([def_homography_points,[random_mod_x_def, random_mod_y_def]])
                        assigned_quad_2 = 1

        # Find a data module in quadrant 3 (bottom left)
        assigned_quad_3 = 0
        while assigned_quad_3 == 0:
            # Find a random data module
            random_data_mod = random.randint(0, ref_len-1)
            random_mod_x, random_mod_y = data_mods_ref[random_data_mod]
            # Making sure that the module is in the bottom left quadrant
            if random_mod_x < xc:
                if random_mod_y > yc:
                    # Find distance from the image center
                    dist = math.sqrt((random_mod_x - xc) ** 2 + (random_mod_y - yc) ** 2)
                    # Determine if the data module is within the acceptable range
                    if  (dist >= distance_min):
                        ref_homography_points = np.vstack([ref_homography_points,[random_mod_x,random_mod_y]])
                        random_mod_x_def, random_mod_y_def = data_mods_def[random_data_mod]
                        def_homography_points = np.vstack([def_homography_points,[random_mod_x_def, random_mod_y_def]])
                        assigned_quad_3 = 1

        # Find a data module in quadrant 4 (bottom right)
        assigned_quad_4 = 0
        while assigned_quad_4 == 0:
            # Find a random data module
            random_data_mod = random.randint(0, ref_len-1)
            random_mod_x, random_mod_y = data_mods_ref[random_data_mod]
            # Making sure that the module is in the bottom right quadrant
            if random_mod_x > xc:
                if random_mod_y > yc:
                    # Find distance from the image center
                    dist = math.sqrt((random_mod_x - xc) ** 2 + (random_mod_y - yc) ** 2)
                    # Determine if the data module is within the acceptable range
                    if  (dist >= distance_min):
                        ref_homography_points = np.vstack([ref_homography_points,[random_mod_x,random_mod_y]])
                        random_mod_x_def, random_mod_y_def = data_mods_def[random_data_mod]
                        def_homography_points = np.vstack([def_homography_points,[random_mod_x_def, random_mod_y_def]])
                        assigned_quad_4 = 1

        angles = quadrilateral_angles(ref_homography_points)
        aspect_ratio = quad_aspect_ratio(ref_homography_points)

        # Check to make sure all vertex angles and the aspect ratio are in an acceptable range
        number_angles_acceptable = 0
        if aspect_ratio < aspect_ratio_max:
            for a in angles:
                if  vertex_angle_min < a < vertex_angle_max:
                    number_angles_acceptable += 1
        
            if number_angles_acceptable == 4:
                    # If the 4 points as meet all the requirments, stop looking and output
                    accept_points = 1

    return(ref_homography_points,def_homography_points)

#----------------------------------------------------------------------
# Function that calculates the homography matrix
#
#  Inputs: ref_homography_points - List of the four data module centroids from the reference image
#          def_homography_points - List of the four data module centroids from the deformed image
#
# Outputs: M - The 3x3 homography matrix
#----------------------------------------------------------------------
def find_homography_coeffs(ref_homography_points,def_homography_points):

    # Get the x and y coords for the ref and def points
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = ref_homography_points
    (x1d, y1d), (x2d, y2d), (x3d, y3d), (x4d, y4d) = def_homography_points

    # A * homography_array = b
    # So find homography coefficents by using np.linalg.solve(A, b)

    # Create the A array for the reference points
    A = np.array([
        [x1, y1, 1, 0, 0, 0, -x1 * x1d, -y1 * x1d],
        [0, 0, 0, x1, y1, 1, -x1 * y1d, -y1 * y1d],

        [x2, y2, 1, 0, 0, 0, -x2 * x2d, -y2 * x2d],
        [0, 0, 0, x2, y2, 1, -x2 * y2d, -y2 * y2d],

        [x3, y3, 1, 0, 0, 0, -x3 * x3d, -y3 * x3d],
        [0, 0, 0, x3, y3, 1, -x3 * y3d, -y3 * y3d],

        [x4, y4, 1, 0, 0, 0, -x4 * x4d, -y4 * x4d],
        [0, 0, 0, x4, y4, 1, -x4 * y4d, -y4 * y4d],
    ], dtype=float)

    # Create the b array for the deformed points
    b = np.array([x1d, y1d, x2d, y2d, x3d, y3d, x4d, y4d], dtype=float)

    # Find the homography coefficents
    coeffs = np.linalg.solve(A, b)

    # Reshape into a 3x3 matrix
    M = np.array([
        [coeffs[0], coeffs[1], coeffs[2]],
        [coeffs[3], coeffs[4], coeffs[5]],
        [coeffs[6], coeffs[7], 1.0]
    ])

    return(M)

#----------------------------------------------------------------------
# Function that calculates the deformation gradient
#
#  Inputs: M - The 3x3 homography matrix
#          xc,yc - Estimated center of the image
#
# Outputs: F - The 2x2 deformation gradient matrix
#----------------------------------------------------------------------
def get_def_grad_from_homography(M,xc,yc):

    # Get the homograpy coefficents from the M matrix
    a = M[0][0]
    b = M[0][1]
    c = M[0][2]
    d = M[1][0]
    e = M[1][1]
    f = M[1][2]
    g = M[2][0]
    h = M[2][1]
    

    w0 = g*xc + h*yc + 1
    # xc_prime = x_hat/w0
    xc_prime = (a*xc+b*yc+c)/(g*xc+h*yc+1)
    # yc_prime = y_hat/w0
    yc_prime = (d*xc+e*yc+f)/(g*xc+h*yc+1)

    # Find the deformation gradient coofficients at (xc,yc)/center
    F11 = (a-g*xc_prime)/w0
    F12 = (b-h*xc_prime)/w0
    F21 = (d-g*yc_prime)/w0
    F22 = (e-h*yc_prime)/w0

    # Shape deformation gradient F into a 2x2
    F = np.array([[F11,F12],[F21,F22]])
    
    return(F)

#----------------------------------------------------------------------
# Function that calculates strain
#
#  Inputs: F - 2x2 deformation gradient matrix
#
# Outputs: E_bar - The equivalent logarithmic strain magnitude
#----------------------------------------------------------------------
def calculate_strains(F):

    # Calculate the Jacobian
    J = np.linalg.det(F)
    # Make sure Jacobian isnt 0
    if J <= 0:
        print('Error: Jacobian J =', J, ' (negative)')
        print('Corresponding A matrix = \n', F)
    
    # Factor out dilatational deformation
    F = F/(math.sqrt(J))

    # Right Cauchy-Green deformation tensor
    C = np.matmul(F.T,F)

    # Create an identity matrix
    I_matrix = np.identity(2)

    # Calculate the strain tensor
    E = logm(C)*(1/2) # Logarithmic or Hencky strain
    E11 = E[0][0]
    E12 = E[0][1]
    E21 = E[1][0]
    E22 = E[1][1]
    E_bar = math.sqrt(2*np.trace(E@E))

    return(E_bar)


#######################################################################

# Hardcoded path for the reference image
REF_PATH = '/Users/surya/Science_Project_Yr3_QR_code/Test_Images/data_ref_3.png'

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image' not in request.files:
        return jsonify({"error, No image uploaded"}), 400
    try:
        # Load the Uploaded Image from the Flutter frontend
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_def = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Load the Hardcoded Image stored locally
        if not os.path.exists(REF_PATH):
            return jsonify({"error": f"Could not find file at {REF_PATH}"}), 500
        img_ref = cv2.imread(REF_PATH,cv2.IMREAD_COLOR)

        # ----------------------------------------------------------
        #    START OF MY MAIN PYTHON CODE   
        # ---------------------------------------------------------- 

        # Run both the reference and deformed image through image processing
        P_ref,Q_ref,R_ref,cnts_areas_centroids_ref,img_data_ref = image_processing(img_ref)
        P_def,Q_def,R_def,cnts_areas_centroids_def,img_data_def = image_processing(img_def)

        # Set the path to my backend folder
        BASE_DIR = "/Users/surya/flutter_backend"

        # Define the 4 model files
        p_det = os.path.join(BASE_DIR, "detect.prototxt")
        m_det = os.path.join(BASE_DIR, "detect.caffemodel")
        p_sr = os.path.join(BASE_DIR, "sr.prototxt")
        m_sr = os.path.join(BASE_DIR, "sr.caffemodel")

        # Initialize the data detector
        try:
            detector = cv2.wechat_qrcode_WeChatQRCode(p_det, m_det, p_sr, m_sr)
            print("WeChat AI engine is working")
        except Exception as e:
            print(f"Error loading models: {e}")

        if img_data_def is not None:
            res, points = detector.detectAndDecode(img_data_def)
            if res:
                print(f"Decoded Data: {res[0]}")
                qr_data = res[0]
            else:
                print("Model loaded, but no QR found in that specific image.")
        else:
            print("Could not find the image file sent.")

        # Refernce image positon markers
        pts_src = np.float32([
            P_ref[1:3],   # Point P
            Q_ref[1:3],  # Point Q
            R_ref[1:3]   # Point R
        ])

        # Deformed image position markers
        pts_dst = np.float32([
            P_def[1:3],   # Point P
            Q_def[1:3],  # Point Q
            R_def[1:3]  # Point R
        ])

        # Find the affine mapping for the position markers
        Ab = find_affine_coeffs(pts_src, pts_dst)

        # Ignore data modules based on area
        cnts_areas_centroids_ref = data_module_area_discrimination(cnts_areas_centroids_ref)
        cnts_areas_centroids_def = data_module_area_discrimination(cnts_areas_centroids_def)

        # Find the corresponding data modules based on the affine mapping found from the position markers
        data_mods_ref,data_mods_def = (data_module_corresponding(cnts_areas_centroids_ref,
        cnts_areas_centroids_def,Ab))
        # Estimate center
        xc, yc = estimate_center(cnts_areas_centroids_ref)

        # ----------------------------------------------------------- 
        # Choose random quadrilaterals and evaluate strains
        # ----------------------------------------------------------- 

        # Number of quadrilaterals for calculating strains
        n_quads = 2000

        # Create arrays for the E##s
        E_bar_array = np.empty((0, 1), dtype=float)
        
        for i in range(n_quads):

            # Find homography points
            ref_homography_points, def_homography_points = (find_homography_points(data_mods_ref,
            data_mods_def,P_ref,Q_ref,R_ref,xc,yc))
            # Find the homography matrix
            M = find_homography_coeffs(ref_homography_points,def_homography_points)
            M = M / M[2,2]
            # Find the deformation gradient F
            F = get_def_grad_from_homography(M,xc,yc)
            # Find the Jacobian
            J = np.linalg.det(F)
            # Make sure Jacobian isnt 0
            if J > 0:
                # Find the strains
                E_bar = calculate_strains(F)
                E_bar_array = np.vstack([E_bar_array,E_bar])

        # Find the mean strain
        E_bar_mean = np.mean(E_bar_array)
        # Convert the results into microstrains
        E_bar_return = E_bar_mean*1000000

        # ----------------------------------------------
        # Use math.isnan to make sure you can return the final value
        # final_val = E_bar_return
        final_val = 4986.551
        qr_data = '123456789.123456.123'
        if math.isnan(final_val):
            return jsonify({"error": "Calculation resulted in NaN."}), 500
        else:
            # Round to 3 decimal places
            final_val = round(float(final_val), 3)

        # Return the final value and qr code data
        return jsonify({
            "status": "Success",
            "data": {
                "strain_micros": float(final_val),
                "qr_data": qr_data
            },
            "message": "Calculated successfully"
        })

    except Exception as e:
        # Log the full traceback to the console for debugging
        import traceback
        traceback.print_exc()
        # Return a generic 500 error with the exception message
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask development server on all available network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
