from tqdm import tqdm
from pointnet_segmentation import Data, DataLoader, classify
from learning3d.models import PointNet, Segmentation, Classifier
from kmeans import cluster, dbscan_cluster
import torch

def point_cloud():
    # License: Apache 2.0. See LICENSE file in root directory.
    # Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

    """
    OpenCV and Numpy Point cloud Software Renderer

    This sample is mostly for demonstration and educational purposes.
    It really doesn't offer the quality or performance that can be
    achieved with hardware acceleration.

    Usage:
    ------
    Mouse: 
        Drag with left button to rotate around pivot (thick small axes), 
        with right button to translate and the wheel to zoom.

    Keyboard: 
        [p]     Pause
        [r]     Reset View
        [d]     Cycle through decimation values
        [z]     Toggle point scaling
        [c]     Toggle color source
        [s]     Save PNG (./out.png)
        [e]     Export points to ply (./out.ply)
        [q\ESC] Quit
    """

    import math
    import time
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    import torch
    from learning3d.models import PointNet, Segmentation, Classifier

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


    state = AppState()

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
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()


    def mouse_cb(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)


    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


    def project(v, out):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


    def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = project(pt1.reshape(-1, 3), out)[0]
        p1 = project(pt2.reshape(-1, 3), out)[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                line3d(out, orig, view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            line3d(out, view(top_left), view(top_right), color)
            line3d(out, view(top_right), view(bottom_right), color)
            line3d(out, view(bottom_right), view(bottom_left), color)
            line3d(out, view(bottom_left), view(top_left), color)

    def render(verts, texcoords, color_source, depth_intrinsics, out): 
        now = time.time()
        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now
        return dt, out

    def pointcloud(out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s], out)
        else:
            proj = project(view(verts), out)

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        proj = np.clip(proj, 0, np.inf)
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        texcoords = np.clip(texcoords, 0, np.inf)
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            # v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint64).T
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]

    out1 = np.empty((h, w, 3), dtype=np.uint8)
    out2 = np.empty((h, w, 3), dtype=np.uint8)
    out3 = np.empty((h, w, 3), dtype=np.uint8)
    out4 = np.empty((h, w, 3), dtype=np.uint8)
    out5 = np.empty((h, w, 3), dtype=np.uint8)
    out6 = np.empty((h, w, 3), dtype=np.uint8)
    out7 = np.empty((h, w, 3), dtype=np.uint8)
    out8 = np.empty((h, w, 3), dtype=np.uint8)
    out9 = np.empty((h, w, 3), dtype=np.uint8)
    out10 = np.empty((h, w, 3), dtype=np.uint8)
    try:
        while True:
            # print('inside while loop')
            # print(f'state: {state.paused}')
            # Grab camera data
            if not state.paused:
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

                if state.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = pc.calculate(depth_frame)
                pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
            # cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9, cluster10 = cluster(verts)
            cluster_labels = dbscan_cluster(verts)
            print('finished clustering')

            clusters = [[] for i in range(np.unique(cluster_labels).shape[0])]
            for i in range(len(verts)):
                pt = verts[i]
                label = cluster_labels[i]
                clusters[label].append(pt)
            print('grouped verts')
            
            outs = []
            for c in clusters:
                o = np.empty((h, w, 3), dtype=np.uint8)
                c = np.array(c)
                dt, o = render(c, texcoords, color_source, depth_intrinsics, o)
                outs.append(o)

            print('len of outs:', len(outs))
            print('num points in cluster1:', len(clusters[0]))
            print('num points in cluster2:', len(clusters[1]))
            cv2.imshow("out0", outs[0])
            cv2.imshow("out1", outs[1])
            # combined_img = np.vstack((outs[0], outs[1]))


            # print(classify(verts))
            # print(classify(cluster2))
              # uv
            # return verts, texcoords

            # # Render
            # dt1, out1 = render(cluster1, texcoords, color_source, depth_intrinsics, out1)
            # dt2, out2 = render(cluster2, texcoords, color_source, depth_intrinsics, out2)
            # dt3, out3 = render(cluster3, texcoords, color_source, depth_intrinsics, out3)
            # dt4, out4 = render(cluster4, texcoords, color_source, depth_intrinsics, out4)
            # dt5, out5 = render(cluster5, texcoords, color_source, depth_intrinsics, out5)
            # dt6, out6 = render(cluster6, texcoords, color_source, depth_intrinsics, out6)
            # dt7, out7 = render(cluster7, texcoords, color_source, depth_intrinsics, out7)
            # dt8, out8 = render(cluster8, texcoords, color_source, depth_intrinsics, out8)
            # dt9, out9 = render(cluster9, texcoords, color_source, depth_intrinsics, out9)
            # dt10, out10 = render(cluster10, texcoords, color_source, depth_intrinsics, out10)

            # # cv2.setWindowTitle(
            #     # state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            #     # (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))
            
            # row1 = np.hstack((out1, out2, out3, out4, out5))
            # row2 = np.hstack((out6, out7, out8, out9, out10))
            # combined_img = np.vstack((row1, row2))
            # cv2.imshow("result", combined_img)
            # cv2.imshow("out5", out5)

            key = cv2.waitKey(1)
            # key = cv2.waitKey(1)

            # cv2.imwrite('./out.png', out)
            time.sleep(5)
    finally:
        # Stop streaming
        pipeline.stop()



def stream():
    ## License: Apache 2.0. See LICENSE file in root directory.
    ## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

    ###############################################
    ##      Open CV and Numpy integration        ##
    ###############################################

    import pyrealsense2 as rs
    import numpy as np
    import cv2

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()


def test():
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    pipeline = rs.pipeline()
    config = rs.config()
    profile = config.resolve(pipeline)
    device = profile.get_device()
    print(f"device: {device}") # MAKE SURE THIS SAYS USB3.0+ OTHERWISE IT WILL NOT WORK
    pipeline.start()
    try:
        frames = pipeline.wait_for_frames()
        for f in frames:
            print(f.profile)
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_image)
        cv2.waitKey(0)
    finally:
        pipeline.stop()

def segment(verts, texcoords): 

    import math
    import time
    import cv2
    import numpy as np
    import pyrealsense2 as rs

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


    state = AppState()

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
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()


    def mouse_cb(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_btns[0] = True

        if event == cv2.EVENT_LBUTTONUP:
            state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = out.shape[:2]
            dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

            if state.mouse_btns[0]:
                state.yaw += float(dx) / w * 2
                state.pitch -= float(dy) / h * 2

            elif state.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                state.translation -= np.dot(state.rotation, dp)

            elif state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                state.translation[2] += dz
                state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            state.translation[2] += dz
            state.distance -= dz

        state.prev_mouse = (x, y)


    cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(state.WIN_NAME, w, h)
    cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj


    def view(v):
        """apply view transformation on vector array"""
        return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


    def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = project(pt1.reshape(-1, 3))[0]
        p1 = project(pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


    def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
                view(pos + np.dot((x, 0, s2), rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
                view(pos + np.dot((s2, 0, z), rotation)), color)


    def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        line3d(out, pos, pos +
            np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        line3d(out, pos, pos +
            np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        line3d(out, pos, pos +
            np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


    def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                line3d(out, orig, view(p), color)
                return p

            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)

            line3d(out, view(top_left), view(top_right), color)
            line3d(out, view(top_right), view(bottom_right), color)
            line3d(out, view(bottom_right), view(bottom_left), color)
            line3d(out, view(bottom_left), view(top_left), color)


    # Classification
    def segment(out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

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


    out = np.empty((h, w, 3), dtype=np.uint8)

    from learning3d.models import Classifier, PointNet, Segmentation
    import torch
    import pyrealsense2 as rs
    import numpy as np

    seg = Segmentation(feature_model=PointNet(), num_classes=40)
    data = np.expand_dims(np.array(verts), axis=0)
    output = seg(torch.from_numpy(data))

    predictions = torch.argmax(output, dim=-1)
    num_classes = 40
    colors = np.random.randint(0, 255, (num_classes, 3))
    color_source = colors[predictions]
    color_source = color_source.astype(np.uint8)


    while True:
        # Render
        now = time.time()

        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            segment(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            segment(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

        cv2.imshow(state.WIN_NAME, out)
        cv2.waitKey(1)


def segment_classifier(verts, texcoords, camera_info):
    from learning3d.models import Classifier, PointNet, Segmentation
    import torch
    # import pyrealsense2 as rs
    import numpy as np
    import cv2
    from torch.utils.data import DataLoader 

    pnet = PointNet(global_feat=False)
    model = Segmentation(feature_model=pnet, num_classes=40)
    model.eval()
    data = np.expand_dims(np.array(verts), axis=0)
    data = torch.from_numpy(data)
    dataset = SegmentationData(data)
    dataloader = DataLoader(dataset=dataset)
    
    for i, pt_data in enumerate(dataloader):
        output = model(pt_data)
    
    predictions = torch.argmax(output, dim=-1)
    print("predictions: " + str(predictions))
    print(predictions.shape)

    # need to project the verts to 2d and then color the 2d verts based on the prediction
    num_classes = 40
    colors = np.random.randint(0, 255, (num_classes, 3))
    color_source = colors[predictions[0]]
    color_source = color_source.astype(np.uint8)
    print(verts.shape)

    h,w = 240, 424
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # w, h = depth_intrinsics.width, depth_intrinsics.height
    # out = np.empty((h, w, 3), dtype=np.uint8)

    def project(v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w

        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj
    proj = project(verts)
    # print(proj.shape)

    painter = True
    # proj now contains 2d image coordinates
    proj = np.clip(proj, 0, np.inf)
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color_source.shape[:2][::-1]
    # if painter:
    #     # sort texcoord with same indices as above
    #     # texcoords are [0..1] and relative to top-left pixel corner,
    #     # multiply by size and add 0.5 to center
    #     v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    # else:
    v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    color_source = color_source.reshape(out.shape)
    print("color_source: " + str(color_source.shape))
    print("u: " + str(u.shape))
    print("v: " + str(v.shape))
    print("out: " + str(out.shape))
    print("i: " + str(i.shape))
    print("j: " + str(j.shape))

    out[i[m], j[m]] = color_source[u[m], v[m]]

    cv2.imshow("image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    


# from torch.utils.data import Dataset

# class SegmentationData(Dataset):
#     def __init__(self, data):
#         self.data = data
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         return self.data[index]

# def classify(verts):
#     pnet = PointNet(global_feat=True)
#     model = Classifier(feature_model=pnet)
#     checkpoint = torch.load('pointnet_segmentation_model.pth', map_location=torch.device('cpu'))  
#     model.load_state_dict(checkpoint)  
#     model.eval()



if __name__ == '__main__':
    point_cloud()