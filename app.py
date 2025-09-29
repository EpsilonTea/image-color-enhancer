import streamlit as st
import torch
from model import ImageEnhancer2
from torchvision import transforms
from torchvision.transforms import ToPILImage
from apply_lut import *
from PIL import Image

# LUTs
luts = get_luts()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
checkpoint_path = './weights/weight.pth'
checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)

model = ImageEnhancer2(luts=luts, output_size=len(luts), mask_num=5)
model.to(device)
model.load_state_dict(checkpoint["model_state_dict"])

transform = transforms.Compose([
        transforms.ToTensor(),
    ])


st.write("## Image Enhancer App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed_image)
        output_image_tensor = output['enhanced_image'].squeeze(0)

    # Convert back to PIL image
    to_pil = ToPILImage()
    enhanced_image = to_pil(output_image_tensor)

    with col2:
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)