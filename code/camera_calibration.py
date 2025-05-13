import cv2
import numpy as np
from tqdm import tqdm

def camera_calibration(img_list):
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
        
    return mtx, dist, board, aruco_dict, arucoParams

def compute_projection_matrix(image, camera_matrix, dist_coeffs, board, aruco_dict, aruco_params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect markers as before
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Estimate pose
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)

    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack((R, tvec))
    projection_matrix = camera_matrix @ extrinsic

    # print(f"projection_matrix: {projection_matrix}")
    return projection_matrix


if __name__ == '__main__':
    compute_projection_matrix()

