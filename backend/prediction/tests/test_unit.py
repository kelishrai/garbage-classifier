from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
from django.urls import reverse
import os
from django.core.files.uploadedfile import SimpleUploadedFile


class APITestCase(TestCase):

    def setUp(self):
        self.client = APIClient()

    def test_get_prediction(self):

        uploaded_file_path = os.path.join(
            os.getcwd(), "prediction/test-images/test.jpg"
        )

        with open(uploaded_file_path, "rb") as file:
            uploaded_file = SimpleUploadedFile(
                name="test.jpg", content=file.read(), content_type="image/jpeg"
            )

        data = {"garbage": uploaded_file}

        response = self.client.post(reverse("get_prediction"), data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        print("test get prediction successful")

    def test_get_plots(self):

        response = self.client.get(reverse("get_plots"))

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        print("test get plot successful")
