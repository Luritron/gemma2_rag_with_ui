# Web-UI for Local LLM using RAG - DevelopsToday Task
## How to run project
1. [Download Ollama](https://ollama.com/download)
2. Download Gemma2 model by running command in cmd: __ollama run gemma2:2b__
3. pip install -r requirements.txt
4. First terminal: __python manage.py runserver__
5. Second terminal: __celery -A base.celery worker --pool=solo -l INFO__
