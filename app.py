import os
from flask import Flask, request, render_template
import torch
from transformers import BartTokenizer
import requests

####### Main program ######
app = Flask(__name__)
# initilization
# init model
bart = torch.load(os.getcwd()+'\\model\\final_model', map_location=torch.device('cpu'))
# init tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


# init page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# serving model and get result
@app.route('/', methods=['POST'])
def index_post():
    # Read the values from the form
    original_text = request.form['text']

    # encode input text
    token_ids = tokenizer(text = original_text, 
                      padding='max_length', truncation=True,return_tensors='pt',max_length=160)
    model_out = bart.generate(token_ids['input_ids'])
    # decoding the generated encoded ids
    translated_text = tokenizer.decode(model_out[0], skip_special_tokens=True)
  
    # Call render template, passing the translated text,
    # original text, and target language to the template
    return render_template(
        'results.html',
        translated_text=translated_text,
        original_text=original_text
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)