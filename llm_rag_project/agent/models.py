from django.db import models

class DialogHistory(models.Model):
    user_id = models.CharField(max_length=255)  # Уникальный идентификатор пользователя
    dialog_id = models.CharField(max_length=255)  # Уникальный идентификатор диалога
    role = models.CharField(max_length=50)  # 'user', 'model', 'system'
    content = models.TextField()  # Текст сообщения
    timestamp = models.DateTimeField(auto_now_add=True)  # Время создания сообщения