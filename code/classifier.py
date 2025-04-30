from learning3d.models import Classifier, PointNet, Segmentation
from depth_camera import point_cloud
import numpy as np
import torch
import cv2
import time

# ptnet = PointNet(emb_dims=1088, use_bn=True)
verts, texcoords = point_cloud()
seg = Segmentation(feature_model=PointNet(), num_classes=40)
data = np.expand_dims(np.array(verts), axis=0)
# print(data.shape)
output = seg(torch.from_numpy(data))
# print("seg shape: " + str(output.shape))
# print("seg: " + str(output))

predictions = torch.argmax(output, dim=-1)
# print("Predictions shape:", predictions.shape)
# print("Predictions (class indices):", predictions)
num_classes = 40
colors = np.random.randint(0, 255, (num_classes, 3))
point_colors = colors[predictions]
point_colors = point_colors.astype(np.uint8)
out = np.empty((h, w, 3), dtype=np.uint8)
while True:
    # Render
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

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    cv2.waitKey(1)


pointcloud(out, verts, texcoords, point_colors)
