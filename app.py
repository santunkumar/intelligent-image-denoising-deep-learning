import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Generator Model (UNet-Based)
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down_block(3, 64, batch_norm=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 512, dropout=True)
        self.up3 = up_block(1024, 512)
        self.up4 = up_block(1024, 256)
        self.up5 = up_block(512, 128)
        self.up6 = up_block(256, 64)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        bottleneck = self.bottleneck(d6)
        
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        
        output = self.final_layer(torch.cat([u6, d1], 1))
        return output

# Load the trained generator
@st.cache_resource
def load_generator():
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

generator = load_generator()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def process(image):
    image = np.array(image)
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return Image.fromarray(denoised)
# Streamlit UI
st.title("🖼️ AI-Powered Image Denoising using GANs")
st.write("Upload a noisy image, and the AI will denoise it for you!")

# File uploader
uploaded_file = st.file_uploader("Upload a noisy image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load Image
    noisy_img = Image.open(uploaded_file).convert("RGB")

    # Preprocess Image
    input_tensor = transform(noisy_img).unsqueeze(0).to(device)
    denoised_img1 = process(noisy_img)
    

    # Generate Denoised Image
    with torch.no_grad():
        denoised_tensor = generator(input_tensor)

    # Postprocess Output
    denoised_tensor = denoised_tensor.squeeze(0).cpu().detach()
    denoised_tensor = denoised_tensor * 0.5 + 0.5  # Convert back to [0,1]
    denoised_img = denoised_tensor.permute(1, 2, 0).numpy()

    # Convert Noisy Image for Display
    noisy_tensor = input_tensor.squeeze(0).cpu().detach()
    noisy_tensor = noisy_tensor * 0.5 + 0.5
    noisy_img_np = noisy_tensor.permute(1, 2, 0).numpy()

    # Display Images
    col1, col2 = st.columns(2)

    with col1:
        st.image(noisy_img_np, caption="Noisy Image", use_column_width=True)

    with col2:
        st.image(denoised_img1, caption="Denoised Image", use_column_width=True)

    st.success("✨ Denoising Completed! You can compare the results above.")
