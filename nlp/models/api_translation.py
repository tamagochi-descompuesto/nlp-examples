import os
import json
import string
import requests
import progressbar
import numpy as np
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu

load_dotenv()

X_RAPID_API_KEY = os.getenv('X_RAPID_API_KEY')
X_RAPID_API_HOST_LIBRE = os.getenv('X_RAPID_API_HOST_LIBRE')
X_RAPID_API_HOST_DEEP = os.getenv('X_RAPID_API_HOST_DEEP')

LIBRE_TRANSLATE_URL = 'https://translate.argosopentech.com/translate'
DEEP_TRANSLATE_URL = 'https://deep-translate1.p.rapidapi.com/language/translate/v2'

LIBRE_HEADERS = {
	"content-type": "application/json",
}
DEEP_HEADERS = {
    "content-type": "application/json",
	"X-RapidAPI-Key": X_RAPID_API_KEY,
	"X-RapidAPI-Host": X_RAPID_API_HOST_DEEP
}
LIBRE_BAR_WIDGETS = ['Making Libre translation:',' [',
         progressbar.Timer(),
         '] ', 
           progressbar.Percentage(),
           progressbar.Bar(),' (',
           progressbar.ETA(), ') '
        ]
DEEP_BAR_WIDGETS = ['Making Deep translation:',' [',
         progressbar.Timer(),
         '] ', 
           progressbar.Percentage(),
           progressbar.Bar(),' (',
           progressbar.ETA(), ') '
        ]

def calc_bleu(original, translation):
    ref = [sentence.split() for sentence in original]
    preprocessed_translation = [sentence.split() for sentence in translation]
    bleu_scores = []

    for sentence in preprocessed_translation:
        bleu_score = sentence_bleu(ref, sentence)
        bleu_scores.append(bleu_score)

    return bleu_scores

def libre_translate(english_data):
    bar = progressbar.ProgressBar(maxval=100, widgets=LIBRE_BAR_WIDGETS).start()
    en_es_translation = []
    i = 0
    for sentence in english_data:
        payload={
            "q": sentence,
            "source": "en",
            "target": "es"
        }
        response = requests.request("POST", LIBRE_TRANSLATE_URL, data=json.dumps(payload), headers=LIBRE_HEADERS)
    
        translation = json.loads(response.text)
        try:
            en_es_translation.append(translation['translatedText'])
            bar.update(i)
            i += 1
        except KeyError:
            print(translation)
    bar.finish()
    return en_es_translation

def deep_translate(english_data):
    bar = progressbar.ProgressBar(maxval=100, widgets=DEEP_BAR_WIDGETS).start()
    en_es_translation = []
    i = 0
    for sentence in english_data:
        payload = {
	        "q": sentence,
	        "source": "en",
	        "target": "es"
        }
        response = requests.request("POST", DEEP_TRANSLATE_URL, json=payload, headers=DEEP_HEADERS)
        
        translation = json.loads(response.text)
        try:  
            en_es_translation.append(translation['data']['translations']['translatedText']) 
            bar.update(i)
            i += 1
        except KeyError:
            print(translation)
    bar.finish()
    return en_es_translation

def global_translate(english_data, spanish_data):
    preprocessed_data_en = []
    preprocessed_data_es = []
    
    for i in range(len(english_data)):
        sentence_en = english_data[i]
        sentence_es = spanish_data[i]

        sentence_en.strip('\n')
        sentence_es.strip('\n')

        for character in string.punctuation:
            sentence_en.replace(character, '')
            sentence_es.replace(character, '')
        
        preprocessed_data_en.append(sentence_en)
        preprocessed_data_es.append(sentence_es)

    libre_translation = libre_translate(preprocessed_data_en)
    deep_translation = deep_translate(preprocessed_data_en)

    libre_bleu_scores = calc_bleu(preprocessed_data_es, libre_translation)
    deep_bleu_scores = calc_bleu(preprocessed_data_es, deep_translation)

    libre_score = np.average(libre_bleu_scores)
    deep_score = np.average(deep_bleu_scores)

    return libre_score, deep_score

if __name__ == '__main__':
    print(global_translate(['Hello, my name is translator', 'I am a very good translator'], ['Hola, mi nombre es traductor', 'Soy un traductor muy bueno']))