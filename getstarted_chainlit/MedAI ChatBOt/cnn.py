import chainlit as cl
import torch
from PIL import Image
import io

# Example: Load a pre-trained image classification model (e.g., ResNet18)
model = torch.hub.load('pytorch/vision:v0.10.0', 'my_model.h5', pretrained=True)
model.eval()

# Preprocess image function (you can adjust according to your model's requirements)
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function to predict the image
def predict_image(image: Image.Image):
    # Preprocess the image
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get predicted class
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

# Chainlit function to handle file upload and image analysis
@cl.on_message
async def main(message: str):
    await cl.Message(content="Please upload an image for analysis.").send()
    
    # Allow the user to upload an image file
    uploaded_file = await cl.FileInput(label="Upload Image", file_type=["image/jpeg", "image/png"])

    # Open the uploaded image file
    image = Image.open(io.BytesIO(uploaded_file.content))

    # Analyze the image using the model
    predicted_class = predict_image(image)
    
    # Send the result back to the user
    await cl.Message(content=f"Predicted class for the uploaded image: {predicted_class}").send()
