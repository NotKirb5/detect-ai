import numpy as np
import pandas as pd
import imgclass

classifier = imgclass.ImageClassifier(img_size=(224,224))
classifier.load_model('detector.keras')
label, confidence = classifier.predict_image('testnsfw/testai.jpg')
print(f"Predicted: {label} (confidence: {confidence:.2%})")
