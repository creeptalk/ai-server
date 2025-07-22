from transformers import pipeline

ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ko")

def translate_ko_to_en(text):
    return ko2en(text)[0]["translation_text"]

def translate_en_to_ko(text):
    return en2ko(text)[0]["translation_text"]
