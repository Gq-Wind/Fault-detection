from django.urls import path
from .views import *

urlpatterns = [
    path('register/', register),
    path('login/', login),
    path('logout/', logout),
    path('refresh_code', refresh_code),
]
