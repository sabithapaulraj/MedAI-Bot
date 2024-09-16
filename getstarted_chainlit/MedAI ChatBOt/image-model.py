import chainlit as cl
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load your trained CNN image detection model
cnn_model = load_model('my_model.h5')  # Path to your trained model

# 2. Preprocess image for CNN model
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize based on your model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Detect objects using the CNN model
def detect_objects(image_path):
    image_array = preprocess_image(image_path)
    predictions = cnn_model.predict(image_array)
    predicted_label = np.argmax(predictions, axis=-1)
    return map_label_to_class(predicted_label)

# 4. Map model output to class names
def map_label_to_class(label):
    # Replace with your actual class mapping
    classes = {0: 'cat', 1: 'dog', 2: 'car', 3: 'tree'}
    return classes.get(label, "Unknown")

# 5. Initialize the Llama 3.1 model
ollama_base_url = "http://test1.dgx.saveetha.in:8080/v1"
chat = ChatOpenAI(base_url=ollama_base_url, openai_api_key="apikey", model="llama3.1")

# 6. Get Llama 3.1 response based on detected object
def get_llama_response(image_label):
    system_message = SystemMessage(content="You are an assistant who responds based on image analysis.")
    user_prompt = f"I detected a {image_label} in the image. Can you provide more information about it?"
    messages = [system_message, HumanMessage(content=user_prompt)]
    
    response = chat.invoke(messages)
    parser = StrOutputParser()
    parsed_response = parser.invoke(response)
    return parsed_response

# 7. Chainlit Form for Uploading Images and Generating Response
@cl.on_chat_start
async def on_chat_start():
    # Prompt to welcome the user
    await cl.Message(content="Welcome to the Image Analysis Assistant! Please upload an image for analysis.").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # This message is only used for text-based interactions
    await cl.Message(content="Please use the form to upload an image.").send()

@cl.on_file_upload
async def handle_file_upload(file: cl.UploadedFile):
    # 1. Save the uploaded file temporarily
    file_path = f"/tmp/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.content)

    # 2. Detect objects using CNN
    detected_object = detect_objects(file_path)
    await cl.Message(content=f"Detected object: {detected_object}").send()

    # 3. Get response from Llama based on the detected object
    llama_response = get_llama_response(detected_object)
    await cl.Message(content=f"Llama Response: {llama_response}").send()

