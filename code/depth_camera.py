from pointnet import classify
from clustering import kmeans_cluster
from mpl_toolkits.mplot3d.axes3d import *
import open3d as o3d
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from sklearn.neighbors import NearestNeighbors

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


class PointCloud:
    def __init__(self, *args, **kwargs):
        self.out = None
        self.state = None
        self.h = None
        self.w = None

    def project(self, v):
            """project 3d vector array to 2d"""
            h, w = self.out.shape[:2]
            view_aspect = float(h)/w

            # ignore divide by zero for invalid depth
            with np.errstate(divide='ignore', invalid='ignore'):
                proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                    (w*view_aspect, h) + (w/2.0, h/2.0)

            # near clipping
            znear = 0.03
            proj[v[:, 2] < znear] = np.nan
            return proj

    def view(self, v):
            """apply view transformation on vector array"""
            return np.dot(v - self.state.pivot, self.state.rotation) + self.state.pivot - self.state.translation

    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(pt1.reshape(-1, 3))[0]
        p1 = self.project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(self, out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(self, out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)
    
    def segment_ground(self, verts, texcoords):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        # Plane segmentation using RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=1000)
        
        mask = np.ones(len(verts), dtype=bool)
        mask[np.array(inliers)] = False

        return verts[mask], texcoords[mask]

    def render(self, out, verts, texcoords, color_source, depth_intrinsics):
        # Render
        now = time.time()

        out.fill(0)
        
        self.grid(out, (0, 0.5, 1), size=1, n=10)
        self.frustum(out, depth_intrinsics)
        self.axes(out, self.view([0, 0, 0]), self.state.rotation, size=0.1, thickness=1)

        if not self.state.scale or out.shape[:2] == (self.h, self.w):
            self.pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(self.state.mouse_btns):
            self.axes(out, self.view(self.state.pivot), self.state.rotation, thickness=4)

        dt = time.time() - now
        return dt, out
    
    def pointcloud(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(v[s])
        else:
            proj = self.project(self.view(verts))

        if self.state.scale:
            proj *= 0.5**self.state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        proj = np.clip(proj, 0, np.inf)
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]

    def voxel_downsampling(self, verts, texcoords):
        # create pointcloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(verts)
        downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=0.02)
        downsampled_points = np.asarray(downsampled_cloud.points)

        # use k nearest neighbors to get the downsampled texcoords
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nbrs.fit(verts)
        down_texcoords = []
        for point in downsampled_points:
            point = point.reshape(1, -1)
            _, indices = nbrs.kneighbors(point)
            down_texcoords.append(texcoords[indices[0][0]])
        
        return downsampled_points, np.array(down_texcoords)

    def visualize_point_cloud(self):
        self.state = AppState()

        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        # Get stream profile and camera intrinsics
        profile = pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        self.w, self.h = depth_intrinsics.width, depth_intrinsics.height

        # Processing blocks
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
        colorizer = rs.colorizer()

        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.state.WIN_NAME, self.w, self.h)

        self.out = np.empty((self.h, self.w, 3), dtype=np.uint8)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_frame = decimate.process(depth_frame)

                # Grab new intrinsics (may be changed by decimation)
                depth_intrinsics = rs.video_stream_profile(
                    depth_frame.profile).get_intrinsics()
                w, h = depth_intrinsics.width, depth_intrinsics.height

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = np.asanyarray(
                    colorizer.colorize(depth_frame).get_data())

                if self.state.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = pc.calculate(depth_frame)
                pc.map_to(mapped_frame)

                # Pointcloud data to arrays
                v, t = points.get_vertices(), points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # verts
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

                # Voxel downsample point cloud
                verts, texcoords = self.voxel_downsampling(verts, texcoords)
                
                # Remove ground from pointcloud data
                verts, texcoords = self.segment_ground(verts, texcoords)

                # Cluster to retrieve pointclouds of objects
                clusters, sub_texcoords = kmeans_cluster(verts, texcoords)
                
                # Classify the object
                classification = classify(clusters[0])
                print(f"Cluster classification is {classification}")

                # Render the pointcloud of the object
                dt, self.out = self.render(self.out, np.array(clusters[0]), np.array(sub_texcoords[0]), color_source, depth_intrinsics)

                cv2.setWindowTitle(
                    self.state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
                    (w, h, 1.0/dt, dt*1000, "PAUSED" if self.state.paused else ""))

                cv2.imshow(self.state.WIN_NAME, self.out)
                cv2.waitKey(1)

                time.sleep(2)
        finally:
            # Stop streaming
            pipeline.stop()


if __name__ == '__main__':
    pt = PointCloud()
    pt.visualize_point_cloud()