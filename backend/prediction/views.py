from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import os
import base64
from .predict import predict_image, preprocess_image
from django.templatetags.static import static
from prediction.serializers import GarbageInputSerializer

# Create your views here.
PREDICTION_MAPPER = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

def index(request):
    return render(request, "index.html")


@api_view(["POST"])
def get_prediction(request):
    serializer = GarbageInputSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    uploaded_file = serializer.validated_data.get("garbage")
    if(uploaded_file is None):
        return JsonResponse({"error": "No file was uploaded"})
    preprocessed_image, file_path = preprocess_image(uploaded_file)
    prediction_result = {}
    prediction_result["resnet50"] = predict_image(
        preprocessed_image,
        "50",
        os.path.join(os.getcwd(), "prediction/output-models/resnet50.pth"),
    )
    prediction_result["resnet18"] = predict_image(
        preprocessed_image,
        "18",
        os.path.join(os.getcwd(), "prediction/output-models/resnet18.pth"),
    )
    prediction_result["resnet34"] = predict_image(
        preprocessed_image,
        "34",
        os.path.join(os.getcwd(), "prediction/output-models/resnet34.pth"),
    )

    return JsonResponse(
        {
            "prediction_result_18": PREDICTION_MAPPER[prediction_result["resnet18"].item()],
            "prediction_result_50": PREDICTION_MAPPER[prediction_result["resnet50"].item()],
            "prediction_result_34": PREDICTION_MAPPER[prediction_result["resnet34"].item()],
            "image": request.build_absolute_uri(static(file_path)),
        }
    )

@api_view(["GET"])
def get_plots(request):
    return JsonResponse(
        {
            "resnet18-plots": {
                "accuracy": request.build_absolute_uri(static('resnet18/accuracy.png')),
                "loss": request.build_absolute_uri(static('resnet18/loss.png')),
                "confusion-matrix": request.build_absolute_uri(static('resnet18/confusion_matrix.png')),
                "loss-batch": request.build_absolute_uri(static('resnet18/loss_batch.png')),
                "lr": request.build_absolute_uri(static('resnet18/lr.png')),
            },
            "resnet50-plots": {
                "accuracy": request.build_absolute_uri(static('resnet50/accuracy.png')),
                "loss": request.build_absolute_uri(static('resnet50/loss.png')),
                "confusion-matrix": request.build_absolute_uri(static('resnet50/confusion_matrix.png')),
                "loss-batch": request.build_absolute_uri(static('resnet50/loss_batch.png')),
                "lr": request.build_absolute_uri(static('resnet50/lr.png')),
            },
             "resnet34-plots": {
                "accuracy": request.build_absolute_uri(static('resnet34/accuracy.png')),
                "loss": request.build_absolute_uri(static('resnet34/loss.png')),
                "confusion-matrix": request.build_absolute_uri(static('resnet34/confusion_matrix.png')),
                "loss-batch": request.build_absolute_uri(static('resnet34/loss_batch.png')),
                "lr": request.build_absolute_uri(static('resnet34/lr.png')),
            }
        }
    )