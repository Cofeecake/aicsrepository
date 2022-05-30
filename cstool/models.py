from django.db import models

# Create your models here.
class Email(models.Model): 
    body = models.TextField(blank=True, null=True)
    prediction = models.TextField(blank=True, null=True)
    
class QuestionAnswer(models.Model): 
    question = models.TextField(blank=True, null=True)
    answer = models.TextField(blank=True, null=True)
    instruction = models.TextField(blank=True, null=True)
    date = models.DateField(auto_now=True)
    active = models.BooleanField(default=True)
    tags = models.TextField(default='other')