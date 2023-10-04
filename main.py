from typing import Optional
from transformers import pipeline
from transformers import BartForConditionalGeneration, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel

origins = [
    'http://localhost:5173'
]
model = BartForConditionalGeneration.from_pretrained('./model/text-sum/')


tokenizer = AutoTokenizer.from_pretrained('./model/text-sum')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Req(BaseModel):
    percentage : Optional[float]=50
    text: str

@app.post('/summarize/')
def getSummary(body:Req):
    arr = body.text.split(' ')
    word_count = len(arr)
    ratio = body.percentage/100
    min_length = int(word_count*ratio)
    max_length = int(word_count*ratio*2)
    print(min_length, max_length)
    summ = pipeline('summarization', model=model, tokenizer=tokenizer, min_length=min_length, max_length = max_length)
    return summ(body.text)