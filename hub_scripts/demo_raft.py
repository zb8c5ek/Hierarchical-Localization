import torch
import numpy as np
import cv2
from torchvision.models.optical_flow import raft_large
import matplotlib.pyplot as plt

# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and move it to CUDA
model = raft_large(pretrained=True).to(device)
model.eval()

# Read two images for computing optical flow
# image1_ori = cv2.imread('E:\develop\Q2Dev2023_windshield_en_verification_pattern\calibrate_intrinsic_and_extrinsic\peculiar_data\p8_dis_1816-lc\check_calib\left/frame_id1000000472_saved_left_img.png')
# image2_ori = cv2.imread('E:\develop\Q2Dev2023_windshield_en_verification_pattern\calibrate_intrinsic_and_extrinsic\peculiar_data\p8_dis_1816-lc\check_calib/right/frame_id1000000472_saved_right_img.png')
# Set scale factor
# s = 0.5  # You can adjust this value

image1_ori = cv2.imread("E:\pgbag\pgbag_2148_x86_64-ubuntu-linux-gcc9.3.0\data\parse/20230613-145438\images\multiperception_cartopic_1_1686639399134.jpg")
image2_ori = cv2.imread("E:\pgbag\pgbag_2148_x86_64-ubuntu-linux-gcc9.3.0\data\parse/20230613-145438\images\multiperception_cartopic_1_1686639399374.jpg")
s = 1.0


# Downsample the images using the scale factor
image1 = cv2.resize(image1_ori, (int(image1_ori.shape[1] * s), int(image1_ori.shape[0] * s))) / 255.0
image2 = cv2.resize(image2_ori, (int(image2_ori.shape[1] * s), int(image2_ori.shape[0] * s))) / 255.0

# Convert BGR to RGB and normalize to [0, 1]
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) / 255.0
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) / 255.0

# Convert images to torch tensors and move to CUDA
image1_tensor = torch.tensor(image1).permute(2, 0, 1).unsqueeze(0).float().to(device)
image2_tensor = torch.tensor(image2).permute(2, 0, 1).unsqueeze(0).float().to(device)
# HD Image Use 14.2 GB GMEM, That's quite ENOUGH.
# Compute optical flow
with torch.no_grad():
    flow_predictions = model(image1_tensor, image2_tensor)
    final_flow = flow_predictions[-1][0].permute(1, 2, 0).cpu().numpy()

# Function to visualize the optical flow
# def visualize_flow(flow):
#     hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
#     hsv[..., 1] = 255
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang * 180 / np.pi / 2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#
# # Visualize the computed flow
# flow_vis = visualize_flow(final_flow)
# plt.imshow(flow_vis)
# plt.axis('off')
# plt.show()

flow = final_flow.copy()
# Function to visualize the optical flow using color coding
def visualize_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
flow_color = visualize_flow(flow)
cv2.imshow('Optical Flow Color', flow_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Function to draw arrows on the optical flow
import matplotlib.cm as cm
def draw_flow_arrows(img, flow, step=40):
    h, w = img.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    mag, _ = cv2.cartToPolar(fx, fy)

    # Normalize the magnitudes to [0,1]
    max_magnitude = np.max(mag)
    normalized_mag = mag / (max_magnitude + 1e-5)  # added small value to avoid division by zero

    # Get colormap from matplotlib's JET colormap
    colormap = cm.jet(normalized_mag)  # This returns RGBA values between 0 and 1
    colormap = (colormap[:, :3] * 255).astype(np.uint8)  # Convert to 0-255 scale and ignore A

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    for i, ((x1, y1), (x2, y2)) in enumerate(lines):
        color = tuple(int(c) for c in colormap[i][0])

        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        cv2.arrowedLine(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2,
                        tipLength=0.2)  # Adjusted thickness and tip length

        # Show magnitude as text
        mag_value = mag[i]
        text_pos = (int(x2), int(y2))
        cv2.putText(img, f"{float(mag_value):.2f}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return img


flow_arrows = draw_flow_arrows(image1.copy(), flow, step=40)

# Display the visualizations using OpenCV

cv2.imshow('Optical Flow Arrows', flow_arrows)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Open3D
import open3d as o3d

# ... [previous code for reading images and computing optical flow]

# Construct 3D point cloud
h, w = flow.shape[:2]
y, x = np.mgrid[0:h, 0:w].astype(np.float32)
x += flow[..., 0]
y += flow[..., 1]

# Create point cloud from the x, y and flow data
xyz = np.zeros((h, w, 3))
xyz[..., 0] = x
xyz[..., 1] = y
xyz[..., 2] = -flow[..., 1]  # use negative for visualization purposes

# Flatten the xyz data
points = xyz.reshape(-1, 3)

# Use the original image color for the point cloud
colors = cv2.cvtColor(image1_ori, cv2.COLOR_BGR2RGB)
colors = colors.reshape(-1, 3) / 255

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Display the point cloud
o3d.visualization.draw_geometries([pcd])
