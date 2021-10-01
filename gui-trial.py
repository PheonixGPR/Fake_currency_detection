from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import ast, os
import cv2
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


model1 = load_model('currency_mobilenetmodel.h5')
f1 = open("mobilenet_currency_class_indices.txt", "r")
labels1 = f1.read()
labels1 = ast.literal_eval(labels1)
final_labels1 = {v: k for k, v in labels1.items()}



model2 = load_model('uv_mobilenetmodel.h5')
f2 = open("mobilenet_uv_class_indices.txt", "r")
labels2 = f2.read()
labels2 = ast.literal_eval(labels2)
final_labels2 = {v: k for k, v in labels2.items()}

def controller(img, brightness,
               contrast):
    if brightness != 0:
  
        if brightness > 0:
  
            shadow = brightness
  
            max = 255
  
        else:
  
            shadow = 0
            max = 255 + brightness
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    # putText renders the specified text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    return cal

def sub(og):
    h, w, c = og.shape
    x_start_r = 0.3125  # 200
    x_end_r = 0.703125  # 450
    y_start_r = 0.270833  # 130
    y_end_r = 0.791666  # 380

    x = int(np.floor(x_start_r * w))
    y = int(np.floor(y_start_r * h))
    x2 = int(np.floor(x_end_r * w))
    y2 = int(np.floor(y_end_r * h))

    image = og[y:y2, x:x2].copy()

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    thresh = image

    return thresh



model3 = load_model('watermark_mobilenetmodel.h5')
f3 = open("mobilenet_watermark_class_indices.txt", "r")
labels3 = f3.read()
labels3 = ast.literal_eval(labels3)
final_labels3 = {v: k for k, v in labels3.items()}



app = Flask(__name__, template_folder = 'templates')
@app.route('/')
def man():
    return render_template("home.html")
@app.route('/',methods = ['GET','POST'])
def home():
	if request.method == 'POST':
		image1 = "Test_images\\" + request.form['a']
		image2 = "Test_images\\" + request.form['b']
		print(image1)
		print(image2)
		x = cv2.imread(image1)
		y = cv2.imread(image2)
		cv2.imshow("1st image",x)
		cv2.waitKey(0)
		cv2.imshow("2st image",y)
		cv2.waitKey(0)
		test_image1 = image.load_img(image1, target_size = (224, 224))
		test_image1 = np.asarray(test_image1)
		test_image1 = np.expand_dims(test_image1, axis=0)
		test_image1 = (2.0 / 255.0) * test_image1 - 1.0
		result1 = model1.predict(test_image1)
		result_dict1 = dict()
		for key in list(final_labels1.keys()):
			result_dict1[final_labels1[key]] = result1[0][key]
		sorted_results1 = {k: v for k, v in sorted(result_dict1.items(), key=lambda item: item[1], reverse=True)}
		final_result1 = dict()
		final_result1[list(sorted_results1.keys())[0]] = sorted_results1[list(sorted_results1.keys())[0]] * 100
		print(final_result1)
		
		
		den = (list(final_result1.keys())[0].split('_'))[0]
		
		original = cv2.imread(image1)
		if den=='10':
			brightness=-130
			contrast=100
		if den=='20':
			brightness=-161
			contrast=67
		if den=='100':
			brightness=-154
			contrast=65
		if den=='500':
			brightness=-154
			contrast=78
		if den=='2000':
			brightness=-150
			contrast=82
		
		cv2.imwrite("input.jpg",controller(original, brightness,contrast))
		z = cv2.imread("input.jpg")
		cv2.imshow("Contrast Adjust",z)
		cv2.waitKey(0)
		test_image2 = image.load_img('input.jpg')
		test_image2 = np.asarray(test_image2)
		test_image2 = np.asarray(sub(test_image2))
		test_image2 = np.expand_dims(test_image2, axis=0)
		test_image2 = (2.0 / 255.0) * test_image2 - 1.0
		result2 = model2.predict(test_image2)
		#print(result2)
		result_dict2 = dict()
		for key in list(final_labels2.keys()):
			result_dict2[final_labels2[key]] = result2[0][key]
		sorted_results2 = {k: v for k, v in sorted(result_dict2.items(), key=lambda item: item[1], reverse=True)}
		final_result2 = dict()
		final_result2[list(sorted_results2.keys())[0]] = sorted_results2[list(sorted_results2.keys())[0]] * 100
		#print(final_result2)
		
		
		
		test_image3 = image.load_img(image2, target_size = (224, 224))
		test_image3 = np.asarray(test_image3)
		test_image3 = np.expand_dims(test_image3, axis=0)
		test_image3 = (2.0 / 255.0) * test_image3 - 1.0
		result3 = model3.predict(test_image3)
		result_dict3 = dict()
		for key in list(final_labels3.keys()):
			result_dict3[final_labels3[key]] = result3[0][key]
		sorted_results3 = {k: v for k, v in sorted(result_dict3.items(), key=lambda item: item[1], reverse=True)}
		final_result3 = dict()
		final_result3[list(sorted_results3.keys())[0]] = sorted_results3[list(sorted_results3.keys())[0]] * 100
		#print(final_result3)
		
		final = []
		final.append(list(final_result2.keys())[0])
		final.append(list(final_result3.keys())[0])
		print(final)
		if (final[0]=='dashed') and final[1]=='yes_watermark':
			return render_template('real.html')
		else:
			return render_template('fake.html')

if __name__ == "__main__":
	app.run(debug=True)
	
	
