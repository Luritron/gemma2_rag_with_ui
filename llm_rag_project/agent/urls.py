from django.urls import path
from .views import api, upload_files

urlpatterns = [
    path("api/", api.urls),
    path("api/upload_files", upload_files),
]
