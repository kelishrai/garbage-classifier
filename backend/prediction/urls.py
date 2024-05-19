from django.urls import path
from .views import index, get_prediction, get_plots
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('', index),
    path('get-prediction', get_prediction, name="get_prediction"),
    path('get-plots', get_plots, name="get_plots")
]+static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
