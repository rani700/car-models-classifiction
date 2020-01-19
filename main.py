from flask import Flask,request
# from flask_cors import CORS
import numpy as np 
import json
from werkzeug import secure_filename

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
loaded_model = load_model('model.h5')




app = Flask(__name__)
# CORS(app)

@app.route('/',methods=['GET' , 'POST'])
def index():
  img_width, img_height = 300, 300
  
  if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      test_image = image.load_img(f.filename, target_size=(img_width, img_height))
      test_image = image.img_to_array(test_image)
      test_image = np.expand_dims(test_image, axis=0)
      test_image = test_image.reshape(1,img_width, img_height,3)
      result = loaded_model.predict(test_image, batch_size=1)
      print(np.argmax(result))
      op = np.argmax(result)
      classes = ["Santro", "Swift", "Wagon R", "i10"]
      return classes[op]
    



if __name__ == "__main__":
  app.run(threaded=False)