import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def camera_calibration():
    file_name0 = glob.glob('../data/extracredit/IMG_3390.jpg') 
    file_name1 = glob.glob('../data/extracredit/IMG_3392.jpg') 
    image0 = cv2.imread(str(file_name0[0]))
    image1 = cv2.imread(str(file_name1[0]))
    # using two of the same image
    img_list = [image0, image1]
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    markerLength = 3.75 # in cm
    # space between markers in the image
    markerSeparation = 0.5 
    board = cv2.aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

    arucoParams = cv2.aruco.DetectorParameters_create()
    counter, corners_list, id_list = [], [], []
    first = True # set to true if it's the first image
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im ,cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
            # a list of the number of aruco markers in each image
            counter.append(len(ids))


    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")

    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    # data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
        
    return mtx, dist, board, aruco_dict, arucoParams

def compute_projection_matrix():
    camera_matrix, dist_coeffs, board, aruco_dict, arucoParams = camera_calibration()
    
    file_name = glob.glob('../data/extracredit/IMG_3390.jpg') 
    image = cv2.imread(str(file_name[0]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect markers as before
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)

    # Estimate pose
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)

    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack((R, tvec))
    projection_matrix = camera_matrix @ extrinsic

    print(f"projection_matrix: {projection_matrix}")
    return projection_matrix


if __name__ == '__main__':
    compute_projection_matrix()

# Checkerboard option: 

# # Set checkerboard size
# CHECKERBOARD = (4, 4) # This is the number of inner corners -> (num cols - 1, num rows - 1) -> needs to be min 3x3 checkerboard with (2,2) corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # Prepare object points (e.g. (0,0,0), (1,0,0), ...)
# objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
# objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)

# objpoints = []  # 3D points in real world
# imgpoints = []  # 2D points in image

# # Load calibration images
# images = glob.glob('../data/extracredit/Checkerboard_pattern.jpg')  # replace with your image folder
# print(os.getcwd())
# print(f"loaded {len(images)} images")

# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
#     if ret:
#         objpoints.append(objp)
#         corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

# # Calibrate
# ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, gray.shape[::-1], None, None
# )

# # Save results
# np.save("camera_matrix.npy", cameraMatrix)
# np.save("dist_coeffs.npy", distCoeffs)

# print("cameraMatrix:\n", cameraMatrix)
# print("distCoeffs:\n", distCoeffs)


