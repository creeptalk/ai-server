from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from app.rag_engine import query_rag_system
# from app.translator import translate_ko_to_en, translate_en_to_ko
from io import BytesIO
# from diffusers import StableDiffusionPipeline
import torch
import openai
import os

from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# pipe = StableDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2",
#     torch_dtype=torch.float16
# ).to("cuda")

class QueryInput(BaseModel):
    query: str
    translate: bool = True

# @app.post("/query")
# def rag_query(input: QueryInput):
#     query = input.query
#     if input.translate:
#         query = translate_ko_to_en(query)
#     response = query_rag_system(query)
#     if input.translate:
#         response = translate_en_to_ko(response)
#     return {"answer": response}

# @app.post("/generate-image")
# def generate_image(input: QueryInput):
#     prompt = input.query
#     if input.translate:
#         prompt = translate_ko_to_en(prompt)

#     image = pipe(prompt, num_inference_steps=25).images[0]
#     image_bytes = BytesIO()
#     image.save(image_bytes, format="PNG")
#     image_bytes.seek(0)

#     return StreamingResponse(image_bytes, media_type="image/png")


@app.post("/test-query")
def gpt_api_query(input: QueryInput):
    query = input.query
    # if input.translate:
    #     query = translate_ko_to_en(query)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    answer = response['choices'][0]['message']['content']

    # if input.translate:
    #     answer = translate_en_to_ko(answer)

    return {"answer": answer}