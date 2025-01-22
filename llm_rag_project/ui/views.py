from django.shortcuts import render

def chat_ui(request):
    return render(request, 'ui/chat.html')
