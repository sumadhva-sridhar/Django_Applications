from django.shortcuts import render

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Create your views here.
def index(request):
	return render(request, "predictor/index.html")

def result(request):
	try:
		tweet = str(request.POST["tweet"])
	except KeyError:
		return render(request, "predictor/error.html", {"message": "Invalid input."})
	if len(tweet) == 0:
		return render(request, "predictor/error.html", {"message": "No input"})

	with open('predictor/trained_model/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	sequences = tokenizer.texts_to_sequences([tweet])
	padded = pad_sequences(sequences, maxlen = 16, padding = 'post', truncating = 'post')
	model = keras.models.load_model("predictor/trained_model/model")
	pos = float(model.predict(padded))
	neg = float(1 - pos)

	return render(request, "predictor/result.html", {"pos": pos, "neg": neg})
