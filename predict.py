
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class cotton:
    def __init__(self,filename):
        self.filename =filename


    def predictioncotton(self):
        # load model
        model = load_model('modelcotton.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if (np.argmax(result) == 0):
            prediction = "Diseased Cotton leaf"
            return [prediction]
        elif (np.argmax(result) == 1):
            prediction = "Diseased cotton plant"
            return [prediction]
        elif (np.argmax(result) == 2):
            prediction = "Cotton leaf is healthy, no treatment is required"
            return [prediction]
        else:
            prediction = "Cotton plant is healthy, no treatment is required"
            return [prediction]


