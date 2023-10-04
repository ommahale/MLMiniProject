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
    min_length : Optional[int]=5
    max_length : Optional[int]=62
    text: str

@app.post('/summarize/')
def getSummary(body:Req):
    summ = pipeline('summarization', model=model, tokenizer=tokenizer, device=0, min_length=body.min_length, max_length = body.max_length)
    return summ(body.text)