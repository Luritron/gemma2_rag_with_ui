from django.shortcuts import render

def chat_ui(request):
    """Рендерит страницу UI для общения с моделью"""
    return render(request, 'ui/chat.html')
