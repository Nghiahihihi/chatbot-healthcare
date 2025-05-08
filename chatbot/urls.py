from django.urls import path


from .views import chat_web, chatbot_predict, reset_session


urlpatterns = [
    path("chatbox/", chat_web),
    path("predict/", chatbot_predict),
    path("reset/", reset_session),
]
