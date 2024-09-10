from django.urls import path
from website.views import *

urlpatterns = [
    path('', home),
    path('embeddings/', embeddings),
    path('services/', services),
    path('all_in_one/', all_in_one),
]