import torch
import random
import requests
import numpy as np
from PIL import Image
import upstash_vector as uv
from dotenv import dotenv_values
from torchvision import transforms
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel

# Load environment variables
env_config = dotenv_values(".env")

# Initialize Upstash Vector Index
index = uv.Index(
    url=env_config["UPSTASH_VECTOR_REST_URL"],
    token=env_config["UPSTASH_VECTOR_REST_TOKEN"]
)
 
# Initialize the CLIP processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Initialize the CLIPModel
image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Initialize the CLIPTextModel
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Image preprocesser
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to fetch remote image
def getImage(url):
    res = requests.get(url, stream=True)
    return Image.open(res.raw)
 
# Function to generate vector embeddings from images
def getImageEmbeddings(image):
    image = preprocess(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = image_model.get_image_features(pixel_values=image)
    embedding = features.squeeze().cpu().numpy()
    return embedding.astype(np.float32)
 
# Function to generate vector embeddings from text
def getTextEmbeddings(text):
    pooled_output = []
    inputs = processor(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = text_model(**inputs)
        pooled_output = text_features.pooler_output
    return pooled_output[0]

# Initialize FastAPI App
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/add")
async def add(request: Request):
    # Load JSON data from the request body
    request_body = await request.json()
    products = request_body['products']
    # Loop over each product
    for product in products:
        embedding = []
        # If the product has an image field
        if product["image"]:
            # Generate vector embeddings of the image of the product
            image = getImage(product["image"])
            embedding = getImageEmbeddings(image)
            # Insert the vector embeddings along with metadata into Upstash Vector
            index.upsert(vectors = [(random.random(), embedding.tolist(), product)])
        if product["title"]:
            # Generate vector embeddings of the title of the product
            text = product["title"]
            embedding = getTextEmbeddings(text)
            # Insert the vector embeddings along with metadata into Upstash Vector
            index.upsert(vectors = [(random.random(), embedding.tolist(), product)])
        if product["description"]:
            # Generate vector embeddings of the description of the product
            description = product["description"]
            embedding = getTextEmbeddings(description)
            # Insert the vector embeddings along with metadata into Upstash Vector
            index.upsert(vectors = [(random.random(), embedding.tolist(), product)])

@app.post("/api/chat")
async def chat(request: Request):
    # Load JSON data from the request body
    request_body = await request.json()
    messages = request_body['messages']
    # Get the latest user message
    user_input = messages[-1]['content']
    query_embedding = []
    # If image URL was passed in the latest user query and get it's vector embeddings
    if "https://" in user_input:
        query_embedding = getImageEmbeddings(getImage(user_input))
    # Else assume text and get it's vector embeddings
    else:
        query_embedding = getTextEmbeddings(user_input)
    # Return the top 3 relevant vectors along with their metadata
    output = index.query(vector=query_embedding.tolist(),  top_k=3, include_metadata=True)
    return [item for item in output if item.score > 0.8]