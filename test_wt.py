import pytest
from transformers import BartForConditionalGeneration,AutoTokenizer, pipeline


def test_func():
    model = BartForConditionalGeneration.from_pretrained('./model/text-sum/')
    assert model!=None

def test_summary():
    model = BartForConditionalGeneration.from_pretrained('./model/text-sum/')
    tokenizer = AutoTokenizer.from_pretrained('./model/text-sum')
    summ = pipeline('summarization', model=model, tokenizer=tokenizer, device=0)
    assert summ is not None

def test_summary_len():
    model = BartForConditionalGeneration.from_pretrained('./model/text-sum/')
    tokenizer = AutoTokenizer.from_pretrained('./model/text-sum')
    summ = pipeline('summarization', model=model, tokenizer=tokenizer, device=0)
    text = 'pytest gathers tests according to a naming convention. By default any file that is to contain tests must be named starting with test_, classes that hold tests must be named starting with Test, and any function in a file that should be treated as a test must also start with test_. If you rename your test file to test_sorts.py and rename the example function you provide above as test_integer_sort, then you will find it is automatically collected and executed. This test collecting behavior can be changed to suit your desires. Changing it will require learning'

    summary = summ(text)
    assert len(text)>len(summary)