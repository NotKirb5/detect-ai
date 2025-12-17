import numpy as np
import pandas as pd
import imgclass

classifier = imgclass.ImageClassifier(img_size=(224,224))
classifier.load_model('my_image_classifier.keras')
label, confidence = classifier.predict_image('testai2.png')
print(f"Predicted: {label} (confidence: {confidence:.2%})")
