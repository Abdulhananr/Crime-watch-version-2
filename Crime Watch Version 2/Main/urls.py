
from django.contrib import admin
from django.urls import path
from home import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('index',views.home,name='index'),
    path('login',views.loginuser,name='login'),
    path('logout',views.logoutuser,name='logout'),
    path('signup', views.signup_user, name='signup'),
    path('contact',views.contact,name='contact'),
    path('FileAudioForensic', views.FileAudioForensic, name='FileAudioForensic'),
    path('LiveAudioForensic', views.LiveAudioForensic, name='LiveAudioForensic'),
    path('upload_file', views.upload_file, name='upload_file'),
    path('test', views.test, name='test'),
    path('LawyerAi', views.LawyerAi, name='LawyerAi'),
    path('Lawyerresponse', views.Lawyerresponse, name='Lawyerresponse'),
    path('testsample', views.testsample, name='testsample')



 


]
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
