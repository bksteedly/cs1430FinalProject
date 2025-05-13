import argparse
import os
from skimage import io
import numpy as np

from helpers import get_matches, show_point_cloud, show_matches
import hw3_code
import camera_calibration


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Homework 3 Camera Geometry")
    parser.add_argument(
        '--sequence',
        required=False,
        default='cards',
        choices=['mikeandikes', 'cards', 'dollar', 'extracredit'],
        help='Which image sequence to use')
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/../data/',
        help='Location where your data is stored')
    parser.add_argument(
        '--ransac-iters',
        type=int,
        default=20,
        help='Number of samples to try in RANSAC')
    parser.add_argument(
        '--num-keypoints',
        type=int,
        default=5000,
        help='Number of keypoints to detect with ORB')
    parser.add_argument(
        '--no-intermediate-vis',
        action='store_true',
        help='Disables intermediate visualizations'
    )
    parser.add_argument(
        '--visualize-ransac',
        action='store_true',
        help="Visualizes Ransac"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = '../data/extracredit/water_bottle'
    image_files = os.listdir(data_dir)

    print(f'Loading files from {data_dir}')
    images = []
    for image_file in image_files:
        images.append(io.imread(os.path.join(data_dir, image_file)))

    print(images[0].shape)
    print('Calibrating camera ...')
    camera_matrix, dist_coeffs, board, aruco_dict, aruco_params = camera_calibration.camera_calibration(images)

    print('Calculating projection matrices...')
    Ms = []
    for image in images:
        M = camera_calibration.compute_projection_matrix(image, camera_matrix, dist_coeffs, board, aruco_dict, aruco_params)
        Ms.append(M)

    
    points3d = []
    points3d_color = []

    for i in range(len(images) - 1):
        image1 = images[i]
        M1 = Ms[i]
        image2 = images[i + 1]
        M2 = Ms[i + 1]

        print(f'Getting matches for images {i + 1} and {i + 2} of {len(images)}...')
        points1, points2 = get_matches(image1, image2, args.num_keypoints)
        if not args.no_intermediate_vis:
            show_matches(image1, image2, points1, points2)

        print(f'Filtering with RANSAC...')
        F, inliers1, inliers2, residual = hw3_code.ransac_fundamental_matrix(
            points1, points2, args.ransac_iters)
        if not args.no_intermediate_vis:
            show_matches(image1, image2, inliers1, inliers2)

        if args.visualize_ransac:
            print(f'Visualizing Ransac')
            hw3_code.visualize_ransac()
            hw3_code.inlier_counts = []
            hw3_code.inlier_residuals = []

        print('Calculating 3D points for accepted matches...')
        points3d_found, inliers1_from3d, inliers1_from3d = hw3_code.matches_to_3d(inliers1, inliers2, M1, M2)
        print(points3d_found)
        points3d += points3d_found.tolist()
        points3d_color += [tuple(image1[int(point[1]), int(point[0]), :] / 255.0) for point in inliers1_from3d]

    points3d = np.array(points3d)

    show_point_cloud(points3d, points3d_color)


if __name__ == '__main__':
    main()
