from django.contrib import admin
from django.urls import path, include
from .views import (
    AICSEndpointView
)

urlpatterns = [
    path('aicsmi', AICSEndpointView.as_view()) 
]
