import cv2 as cv
import numpy as np

dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)

num_markers_x = 3
num_markers_y = 3
marker_length = 3.75
marker_separation = 0.5

marker_ids = [0, 9, 18, 27]

for i in range(4):
    board = cv.aruco.GridBoard_create(
        num_markers_x,              # Number of markers in the X direction.
        num_markers_y,              # Number of markers in the Y direction.
        marker_length,              # Length of the marker side.
        marker_separation,          # Length of the marker separation.
        dictionary,                  # The dictionary of the markers.
        marker_ids[i]               # (optional) Ids of all the markers (X*Y markers).
    )

    img = np.zeros((580,725), np.uint8) #4x4

    img = board.draw( 
        img.shape,                  # Size of the output image in pixels.
        img,                        # Output image with the board
        0,                          # Minimum margins (in pixels) of the board in the output image
        1                           # Width of the marker borders
    )

    extension = ".jpg"
    cv.imwrite("aruco_" +
                str(i) + "_" + 
                str(num_markers_x) + "x" + str(num_markers_y) + "_" +
                str(int(marker_length*100)) + "cm_length_" +
                str(int(marker_separation*100)) + "cm_space"
                + extension, img)
    cv.imshow("aruco_board", img)
    cv.waitKey(0)