from django.urls import path
from django.conf.urls import url, include

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("game", views.game, name="game"),
    path("question", views.question, name="question"),
    url(r'^input$', views.get_canvas)
]
