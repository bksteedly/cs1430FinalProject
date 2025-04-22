import cv2
import numpy as np
import glob
import os


file_name = glob.glob('../data/extracredit/multiple_aruco.jpg') 
images = cv2.imread(str(file_name[0]))
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
markerLength = 10 # in cm
markerSeparation = 0.5 
board = cv2.aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)
img = board.draw((864,1080))
cv2.imshow("aruco", img)

arucoParams = cv2.aruco.DetectorParameters_create()
counter, corners_list, id_list = [], [], []
first = True
img_gray = cv2.cvtColor(images ,cv2.COLOR_RGB2GRAY)
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
if first == True:
    corners_list = corners
    id_list = ids
    counter = [1]
    first = False
else:
    corners_list = np.vstack((corners_list, corners))
    id_list = np.vstack((id_list,ids))
    counter.append(len(ids))

# markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
#         images, aruco_dict, parameters=arucoParams)
# # markerIds = [m[0] for m in markerIds]
# # markerCorners = [m[0] for m in markerCorners]
# corners_list = markerCorners
# id_list = markerIds
# counter = [1]


counter = np.array(counter)
print ("Calibrating camera .... Please wait...")

print(corners_list)
print(id_list)
print(counter)
print(board)
print(img_gray.shape)
ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    
cv2.destroyAllWindows()

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


