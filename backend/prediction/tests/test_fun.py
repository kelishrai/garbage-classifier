import unittest
import torch
import os
from PIL import Image
from prediction.predict import preprocess_image, predict_image

class TestPredictionFunctions(unittest.TestCase):

    def setUp(self):
        self.image_path =  os.path.join(os.getcwd(), "prediction/test-images/test.jpg")
        self.model_name = "18"  
        self.model_path = os.path.join(os.getcwd(), "prediction/output-models/resnet18.pth")
        self.preprocessed_image = preprocess_image(self.image_path)

    def test_preprocess_image(self):
        self.assertIsInstance(self.preprocessed_image, torch.Tensor)
        print("test preprocess image successful")

    def test_predict_image(self):
        predicted_class = predict_image(self.preprocessed_image, self.model_name, self.model_path)
        self.assertIsInstance(predicted_class.item(), int)
        print(predicted_class)

        print("test prediction image successful")
