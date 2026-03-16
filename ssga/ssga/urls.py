"""
URL configuration for ssga project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from innovation import views

# urls.py (in your app or project)
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.home, name="home"),
    path("soil_im_predict/", views.soil_im_predict, name="soil_im_predict"),
    path("fertilizer/", views.fertilizer, name="fertilizer"),
    path("routinetrack4/", views.routinetrack4, name="routinetrack4"),
    path("croprecom3/", views.croprecom3, name="croprecom3"),
    path("index/", views.index, name="soil_index"),
    path("pltdis/", views.pltdis, name="pltdis"),
    path("growth/", views.growth, name="growth"),
    path("gov/", views.gov, name="gov"),
    path("aiyield/", views.aiyield, name="aiyield"),
    path("advisory5/", views.advisory5, name="advisory5"),
    path("soil_image_predict/", views.soil_image_predict, name="soil_image_predict"),
    path("soil_predictor/", views.soil_predictor, name="soil_predictor"),
    path("soiledu/", views.soiledu, name="soiledu"),
    path("wastedis/", views.wastedis, name="wastedis"),
]

