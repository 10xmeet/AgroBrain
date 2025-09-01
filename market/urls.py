from django.urls import path
from .views import market_view

urlpatterns = [
    path('', market_view, name="market"),
]
