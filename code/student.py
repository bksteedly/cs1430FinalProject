import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from camera_calibration import compute_projection_matrix

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    # dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    # parameters = cv2.aruco.DetectorParameters_create()

    # markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
    #     image, dictionary, parameters=parameters)
    # markerIds = [m[0] for m in markerIds]
    # markerCorners = [m[0] for m in markerCorners]

    # # image = cv2.imread("/Users/amulya/Documents/Brown/spring_2025/csci1430/cs1430FinalProject/data/extracredit/Checkerboard_pattern.jpg")
    # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # print(gray)
    # # pattern_size = (4,4)
    # # print(pattern_size)

    # # ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # # print(ret)
    # # print(corners)

    # # if ret:
    # #     cv2.drawChessboardCorners(image, pattern_size, corners, ret)
    # #     cv2.imshow('Chessboard Corners', image)
    # #     cv2.waitKey(0)
    # #     cv2.destroyAllWindows()
    # # else:
    # #     print("Chessboard pattern not found.")

    # points2d = []
    # points3d = []

    # for markerId, marker in zip(markerIds, markerCorners):
    #     if markerId in markers:
    #         for j, corner in enumerate(marker):
    #             points2d.append(corner)
    #             points3d.append(markers[markerId][j])

    # points2d = np.array(points2d)
    # points3d = np.array(points3d)

    # ########################
    # # TODO: Your code here #
    # ########################
    # num_points = points2d.shape[0]
    # A = []
    # b = []
    
    # for i in range(num_points):
    #     [u, v] = points2d[i]
    #     [X, Y, Z] = points3d[i]

    #     A.append([X, Y, Z, 1, 0, 0, 0, 0, -X*u, -Y*u, -Z*u])
    #     A.append([0, 0, 0, 0, X, Y, Z, 1, -X*v, -Y*v, -Z*v])
    #     b.append([u])
    #     b.append([v])

    # A = np.array(A)
    # b = np.array(b)


    # M, residual, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # M = np.append(M, 1)
    # M = M.reshape((3, 4))

    # return M, residual

    return compute_projection_matrix(), None

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform a point into 
    a line within the second image - the epipolar line - such that F x' = l. 
    Fitting a fundamental matrix to a set of points will try to minimize the 
    error of all points x to their respective epipolar lines transformed 
    from x'. The residual can be computed as the difference from the known 
    geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared error in the estimation
    """
    ########################
    # TODO: Your code here #
    ########################

    A = []
    for i in range(points1.shape[0]):
        u, v = points1[i][0], points1[i][1]
        u_prime, v_prime = points2[i][0], points2[i][1]
        
        A.append([u*u_prime, v*u_prime, u_prime, u*v_prime, v*v_prime, v_prime, u, v, 1])
    A = np.array(A)

    _, _, Vh = np.linalg.svd(A)
    F = Vh[-1]
    F = F.reshape((3,3))

    U, S, Vh = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vh

    points1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_homogeneous = np.hstack((points2, np.ones((points2.shape[0], 1))))
    residual = np.sum(np.abs(np.sum(np.dot(points2_homogeneous, F) * points1_homogeneous, axis=1)) ** 2)

    return F, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points. See the handout for a detailing of the RANSAC method.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the sum of the square error induced by best_Fmatrix upon the inlier set

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'
    num_points = matches1.shape[0]
    threshold = 1e-2 
    best_Fmatrix = np.zeros(9).reshape((3, 3))
    best_inliers_a = []
    best_inliers_b = []
    best_inlier_residual = 0 

    for iter in range(num_iters):
        point_idx = np.random.choice(num_points, size=8, replace=False) 
        small_points1_array = matches1[point_idx]
        small_points2_array = matches2[point_idx]
        
        # F, _ = cv2.findFundamentalMat(small_points1_array, small_points2_array, cv2.FM_8POINT, 1e10, 0, 1)
        F, _ = estimate_fundamental_matrix(small_points1_array, small_points2_array)

        inliers_a = []
        inliers_b = []
        residual = 0
        for i in range(num_points):
            x1 = np.append(matches1[i], 1)
            x2 = np.append(matches2[i], 1)

            r = np.abs(np.dot(np.dot(x2.T, F), x1))

            if r < threshold:
                inliers_a.append(matches1[i])
                inliers_b.append(matches2[i])
                residual += r**2

        inlier_counts.append(len(inliers_a))
        inlier_residuals.append(residual)

        if len(inliers_a) > len(best_inliers_a):
            best_Fmatrix = F
            best_inliers_a = inliers_a
            best_inliers_b = inliers_b
            best_inlier_residual = residual
    
    best_inliers_a = np.array(best_inliers_a)
    best_inliers_b = np.array(best_inliers_b)
    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual

def matches_to_3d(points2d_1, points2d_2, M1, M2, threshold=1.0):
    """
    Given two sets of corresponding 2D points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    You may find that some 3D points have high residual/error, in which case you 
    can return a subset of the 3D points that lie within a certain threshold.
    In this case, also return subsets of the initial points2d_1, points2d_2 that
    correspond to this new inlier set. You may modify the default value of threshold above.
    All local helper code that calls this function will use this default value, but we
    will pass in a different value when autograding.

    N is the input number of point correspondences
    M is the output number of 3D points / inlier point correspondences; M could equal N.

    :param points2d_1: [N x 2] points from image1
    :param points2d_2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :param threshold: scalar value representing the maximum allowed residual for a solved 3D point

    :return points3d_inlier: [M x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points2d_1 and points2d_2
    :return points2d_1_inlier: [M x 2] points as subset of inlier points from points2d_1
    :return points2d_2_inlier: [M x 2] points as subset of inlier points from points2d_2
    """
    ########################
    # TODO: Your code here #

    points3d_inlier = []
    points2d_1_inlier = []
    points2d_2_inlier = []

    # Solve for ground truth points
    num_points = points2d_1.shape[0]

    for i in range(num_points):
        x1 = points2d_1[i]
        x2 = points2d_2[i]

        A = np.array([
            [x1[0]*M1[2,0]-M1[0,0], x1[0]*M1[2,1]-M1[0,1], x1[0]*M1[2,2]-M1[0,2]],
            [x1[1]*M1[2,0]-M1[1,0], x1[1]*M1[2,1]-M1[1,1], x1[1]*M1[2,2]-M1[1,2]],
            [x2[0]*M2[2,0]-M2[0,0], x2[0]*M2[2,1]-M2[0,1], x2[0]*M2[2,2]-M2[0,2]],
            [x2[1]*M2[2,0]-M2[1,0], x2[1]*M2[2,1]-M2[1,1], x2[1]*M2[2,2]-M2[1,2]]
        ])
        x = np.array([[M1[0,3]-M1[2,3]*x1[0]], [M1[1,3]-M1[2,3]*x1[1]], [M2[0,3]-M2[2,3]*x2[0]], [M2[1,3]-M2[2,3]*x2[1]]])
        X, residual, _, _ = np.linalg.lstsq(A, x, rcond=None) # TODO: use the residual to threshold
        # if residual[0] < threshold:
        X = X.reshape((3,))
        points3d_inlier.append(X)
        points2d_1_inlier.append(x1)
        points2d_2_inlier.append(x2)


    ########################

    points3d_inlier = np.array(points3d_inlier)
    points2d_1_inlier = np.array(points2d_1_inlier)
    points2d_2_inlier = np.array(points2d_2_inlier)
    
    return points3d_inlier, points2d_1_inlier, points2d_2_inlier


#/////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()
