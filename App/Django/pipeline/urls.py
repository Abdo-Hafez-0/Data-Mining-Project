from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('cleaning/', views.cleaning_page, name='cleaning'),
    path('eda/', views.eda_page, name='eda'),
    path('features/', views.features_page, name='features'),
    path('modeling/', views.modeling_page, name='modeling'),
]
