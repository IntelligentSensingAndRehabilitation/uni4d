import torch
import os
import cv2
from PIL import Image
import json
from dotenv import load_dotenv

import torchvision.transforms as TS

from ram.models import ram
from ram import inference_ram as inference

from openai import OpenAI
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def get_chatgpt_response(prompt):
    try:
        response = client.chat.completions.create(
            # messages=[
            #     {"role": "system", "content": "You are a data preprocessing assistant that will filter out key words from a list of words. \
            #      You will be given a list of words and you will need to filter out words that belong to living things that are able to move \
            #      by itself. Make sure that you go over every single word and explain if they can move and is living, explaining your thought process. \
            #      Return your response as only a json, with the reasoning for each word under the key 'reasoning' and the words that are living and can move under the key 'dynamic'."},
            #     {"role": "user", "content": f"{prompt}"}
            # ],
            messages=[
                {"role": "system", "content": "You are a data preprocessing assistant that will filter out key words from a list of words. \
                 You will be given a list of words and you will need to filter out words that are nouns are is able to move by itself. \
                 Make sure that you go over every single word and explain if the word is a noun, and whether it is able to move. Explain your thought process. \
                 Return your response as only a json, with the reasoning for each word under the key 'reasoning' and the words that are nouns and can move by itself under the key 'dynamic'."},
                {"role": "user", "content": f"{prompt}"}
            ],
            model="gpt-4o",
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(str(e))
        return str(e)

def load_model(model_path,image_size=384):

    if load_model.model is None:
        print(f"Loading model from {model_path}...")
        load_model.model = ram(
            image_size=image_size,
            pretrained=model_path,
            vit='swin_l'
        )
        load_model.model.eval()
        load_model.model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Model loaded successfully.")
    return load_model.model

load_model.model = None


def process_capture(capture, model, device="cuda"):

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
                    TS.Resize((384, 384)),
                    TS.ToTensor(), normalize
                ])

    all_tags = set()

    # iterate through all images in the video
    while True:
        ret, frame = capture.read()

        if not ret:
            # Break the loop if there are no frames left or error reading frame
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_pil = Image.fromarray(frame_rgb)

        raw_image = image_pil.resize(
                        (384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(device)

        res = inference(raw_image , model)

        all_tags.update(set(res[0].split(' | ')))

    response = get_chatgpt_response(f"{list(all_tags)}")
    response['input'] = list(all_tags)
    response['dynamic'] = list(set(response['dynamic']))               # remove duplicates

    return response



