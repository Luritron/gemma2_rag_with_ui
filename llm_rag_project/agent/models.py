from django.db import models

class DialogHistory(models.Model):
    user_id = models.CharField(max_length=255)
    dialog_id = models.CharField(max_length=255)
    role = models.CharField(max_length=50)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)