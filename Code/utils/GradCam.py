import torch
import torch.nn as nn
# import torchvision.models as models
from Code.material.CSTNet import CST_Net
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer = self.get_target_layer()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def get_target_layer(self):
        if isinstance(self.target_layer, str):
            modules = dict([*self.model.named_modules()])
            if self.target_layer not in modules:
                raise ValueError(f"Invalid layer name: {self.target_layer}")
            return modules[self.target_layer]
        elif isinstance(self.target_layer, int):
            return list(self.model.modules())[self.target_layer]
        else:
            raise ValueError("Invalid layer type. Layer should be either a string or an integer.")

    def get_gradients(self):
        return self.gradients

    def get_feature_maps(self):
        return self.feature_maps

    def __call__(self, x):
        device = next(self.model.parameters()).device
        input_tensor = x.to(device)

        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Backward pass
        one_hot = torch.zeros_like(output, dtype=torch.float)
        one_hot[0][output.argmax()] = 1
        output.backward(gradient=one_hot)

        # Retrieve gradients and feature maps
        gradients = self.get_gradients()[0]
        feature_maps = self.get_feature_maps()[0]

        # Compute weights using global average pooling
        weights = torch.mean(gradients, axis=(1, 2), keepdims=True)

        # Compute cam
        cam = torch.sum(weights * feature_maps, axis=0)
        cam = np.maximum(cam, 0)  # ReLU function
        cam = cam / torch.max(cam)  # Normalize

        # Resize cam
        cam = cv2.resize(cam.cpu().numpy(), (x.shape[2], x.shape[3]))

        return cam

# Load pre-trained model
model = CST_Net(channel=32,n_class=1) # 换成自己的网络

# Specify target layer# in the model
target_layer = "resnet" # 指定使用网络中的哪一层

# Create GradCam object
grad_cam = GradCam(model, target_layer)

# Load image
img_path = "/home/stu/zy/data/Kvasir/Kvasir-SEG/images/cju2top2ruxxy0988p1svx36g.jpg" # 输入一张图片
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img_tensor = torch.tensor(img, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)

# Generate Grad-CAM
cam = grad_cam(img_tensor)

# Plot original image and Grad-CAM
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax1.axis("off")
ax2.imshow(img)
ax2.imshow(cam, cmap="jet", alpha=0.5)
ax2.axis("off")
plt.show()

# Save CAM image。 分辨率为300 边界框为tight
plt.savefig("/home/stu/zy/data/CSTNet CAM/cam03.png", dpi=300, bbox_inches="tight")

plt.show()