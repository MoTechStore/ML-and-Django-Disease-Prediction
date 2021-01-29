from django.urls import path, include
from . import views


urlpatterns = [
 path('', views.home, name="home"),
 path('predict', views.predictdisease, name="predictdisease"),
 path('team', views.team, name="team"),
 path('predict', views.predict, name="predict"),
 path('about', views.about, name="about"),

]