
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import HttpResponse, JsonResponse
from datetime import datetime, timezone
from .models import Email, QuestionAnswer
import json
import datetime
from cstool.deployment.make_pred import predict_email_content

class CSRFExemptMixin(object):
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super(CSRFExemptMixin, self).dispatch(*args, **kwargs)

class AICSEndpointView(CSRFExemptMixin, View): # AICS addon link
    def post(self, request):
        # Grabbing and formatting request data 
        request_body = request.body
        request_body = str(request_body).split("'", 2)
        request_dict = json.loads(request_body[1])
        # Checks to see if the post request is looking to be predicted or already has a tag
        if 'tag' in request_dict:
            # Creates a simple empty list to be filled with dictionarys of knowedge base entries 
            info = []
            # Searches through the knowedge base and grabs any entry with at least one tag matching the requested tag
            for index in list(QuestionAnswer.objects.values()):
                tags = str(index['tags'])
                if str(request_dict['tag']) in tags.lower():
                    # Adds the knowledge base entry into the empty list
                    info.append(index)
            return JsonResponse(info, safe=False)
        elif 'predict' in request_dict:
            # Predicts the topic of the body of the email data
            pdct = str(predict_email_content(request_dict['body']))
            request_dict['prediction'] = pdct
            # Filters out all the extra characters that get added to the prediction by the model 
            bad_characters = ["'", "[", "]"]
            prediction = str(request_dict['prediction'])
            for character in bad_characters:
                prediction = prediction.replace(character, '')
            info = []
            # Searches through the knowedge base and grabs any entry with at least one tag matching the prediction
            for index in list(QuestionAnswer.objects.values()):
                tags = str(index['tags'])
                if str(prediction) in tags.lower():
                    info.append(index)
            # adds the email data and prediction to the beginning of the list        
            info.insert(0, request_dict)
            Email(body=request_dict['body'], prediction=request_dict['prediction']).save()
            return JsonResponse(info, safe=False)
        elif 'create' in request_dict:
            QuestionAnswer(question=request_dict['question'], tags=request_dict['tags'], answer=request_dict['answer'], instruction=request_dict['instruction'], active=True).save()
            return HttpResponse('Created')
        elif 'edit_id' in request_dict:
            QuestionAnswer.objects.get(pk = request_dict['edit_id'])
            if 'question' in request_dict:
                QuestionAnswer(question = request_dict['question']).save()
            if 'answer' in request_dict:
                QuestionAnswer(answer = request_dict['answer']).save()
            if 'tags' in request_dict:
                QuestionAnswer(tags = request_dict['tags']).save()
            if 'instruction' in request_dict:
                QuestionAnswer(instruction = request_dict['instruction']).save()
            if 'active' in request_dict:
                QuestionAnswer(active = request_dict['active']).save()
            QuestionAnswer(date = datetime.date.today()).save()
            return HttpResponse('Edited')  

    def get(self, request):
        info = []
            # Searches through the knowedge base and grabs any entry with at least one tag matching the prediction
        for index in list(QuestionAnswer.objects.values()):
            info.append(index)
            # adds the email data and prediction to the beginning of the list        
        return JsonResponse(info, safe=False)