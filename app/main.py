from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.rag_engine import query_rag_system
from app.translator import translate_ko_to_en, translate_en_to_ko
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

# pipe = StableDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2",
#     torch_dtype=torch.float16
# ).to("cuda")

# class QueryInput(BaseModel):
#     query: str
#     translate: bool = True

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


class HelloInput(BaseModel):
    message: str

@app.post("/hello")
def hello_endpoint(input: HelloInput):
    return {"response": f"You said: {input.message}"}