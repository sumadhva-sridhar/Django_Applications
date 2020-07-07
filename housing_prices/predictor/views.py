from django.shortcuts import render

import numpy as np
from tensorflow import keras

# Create your views here.
def index(request):
	return render(request, "predictor/index.html")

def result(request):
	try:
		value = float(request.POST["val"])
	except KeyError:
		return render(request, "predictor/error.html", {"message": "Invalid input."})

	model = keras.models.load_model("predictor/my_model")
	r = [value]
	result = float(model.predict(r))

	return render(request, "predictor/result.html", {"result": result})
