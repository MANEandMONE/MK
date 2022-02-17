from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ver1', views.ver1, name='ver1'),
    path('ver3', views.ver3, name='ver3'),
    # path('vr0', views.vr0, name='vr0')
    # path('result1', views.result1, name = 'result1')
    # 회원가입
    #path('register/',views.register),
    #path('ver2', views.ver2, name = 'ver2'),
]