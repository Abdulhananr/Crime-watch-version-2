from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import logout, authenticate, login
from datetime import datetime
import json,random
from django.db import IntegrityError
from home.models import Contact
from django.contrib import messages
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import logout, authenticate, login
from datetime import datetime
from home.models import Contact
from django.http import JsonResponse
import os
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.stem import WordNetLemmatizer
import nltk
import requests

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
nltk.download('wordnet')
os.environ['OPENAI_API_KEY'] =  ""

import openai
# Load NLTK WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the fine-tuned model and tokenizer (replace 'fine_tuned_model' with actual path)
model_name = r"C:\Users\dell\OneDrive\Desktop\New folder (2)\fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\dell\OneDrive\Desktop\New folder (2)\fine_tuned_model\tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def stringify_list_items(my_list):
    if not my_list:  # Check if the list is empty
        return 'NONE'  # Return 'NONE' if the list is empty
    else:
        return ''.join(str(item) if item != '' else 'NONE' for item in my_list)

# Function to lemmatize a word
def lemmatize_word(word):
    return lemmatizer.lemmatize(word, pos='v')  # 'v' indicates that the word is a verb

# Function to predict whether text is criminal or not and identify criminal words
def predict_and_identify_criminal(text):
    # Prepend [CLS] token and append [SEP] token
    text = "[CLS] " + text + " [SEP]"

    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

    # Perform forward pass through the model
    outputs = model(**inputs)

    # Get the predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Tokenize input text to get word indices
    tokens = tokenizer.tokenize(text)
    word_indices = tokenizer.convert_tokens_to_ids(tokens)

    # Identify words related to criminal activity
    criminal_words = ['abducted', 'abuse', 'ambush', 'break', 'dacoity', 'devise', 'devised', 'dispose', 'divert', 'fight', 'fire', 'fired', 'firing', 'fraud', 'gun', 'hide', 'kidnap', 'kidnapped', 'kidnapping', 'kill', 'killed', 'killing', 'loot', 'looted', 'murder', 'murdered', 'murdering', 'poisoned', 'raid', 'raided', 'rob', 'robbed', 'robbery', 'robbing', 'sexual', 'shoot', 'smuggled', 'smuggling', 'snatch', 'snatched', 'snatching', 'steal', 'stealing', 'stole', 'stolen', 'theft','threatened']

    # Lemmatize criminal words
    lemmatized_criminal_words = [lemmatize_word(word) for word in criminal_words]

    # Check if any criminal words are present in the input text
    criminal_present = any(token.lower() in lemmatized_criminal_words for token in tokens)

    # Construct output text with identified words
    output_text = ""
    criminal_found = []
    for i, token in enumerate(tokens):
        if word_indices[i] == tokenizer.cls_token_id:
            continue
        if word_indices[i] == tokenizer.sep_token_id:
            break
        # Lemmatize the token for comparison
        lemma_token = lemmatize_word(token.lower())
        if lemma_token in lemmatized_criminal_words:
            output_text += f"**{token}** "
            criminal_found.append(token)
        else:
            output_text += token + " "

  
    # Return prediction along with input text, identified words, locations, dates, and times
    prediction_result = {
        "prediction": "Detected" if predicted_class == 1 else "Not Detected.",
        "criminal_words": criminal_found,
        
    }
    return prediction_result
def LawyerAi(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/login") 
    
    return render(request,'LawyerAi.html',{'username': username})

def home(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/login") 
    return render(request,'index.html',{'username': username})
def loginuser(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, 'login.html')

    return render(request, 'login.html')
def logoutuser(request):
    logout(request)
    return redirect("/login")

def index(request):
    username = request.user.username if request.user.is_authenticated else None
    if not username:
        return redirect("/login") 

    apiUrl = "https://newsapi.org/v2/everything?q=criminal&from=2024-05-05&sortBy=publishedAt&apiKey=f07b0882d4a94064a1f0a5214a44f427"
    response = requests.get(apiUrl)
    data = response.json()

    # Check if articles are available in the response
    if 'articles' in data:
        # Randomly select 3 articles
        articles = data['articles']
        random_articles = random.sample(articles, min(len(articles), 3))
    else:
        random_articles = []

    # Pass the articles to the template
    return render(request, 'index.html', {'articles': random_articles, 'username': username})




def signup_user(request):
    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        try:
            myuser = User.objects.create_user(username, email, password)
            myuser.first_name = ""
            myuser.last_name = ""
            myuser.save()
            messages.success(request, "Account created successfully. Please log in.")
            return redirect('/login')
        except IntegrityError:
            messages.error(request, "Username or email already exists")
            return render(request, 'login.html')

    return render(request, 'login.html')
def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        desc = request.POST.get('desc')
        contact = Contact(name=name, email=email, phone=phone, desc=desc, date = datetime.today())
        contact.save()
    
    username = request.user
    return render(request,'index.html',{'username': username})
def test(request):
    # Fetch data from an API or any other source
    apiUrl = "https://newsapi.org/v2/everything?q=healthcare&from=2024-05-05&sortBy=publishedAt&apiKey=f07b0882d4a94064a1f0a5214a44f427"
    response = requests.get(apiUrl)
    data = response.json()

    # Check if articles are available in the response
    if 'articles' in data:
        # Randomly select 10 articles
        articles = data['articles']
        random_articles = random.sample(articles, min(len(articles), 30))
        data['articles'] = random_articles
    else:
        data['articles'] = []
    username = request.user

    data_json = json.dumps(data)

    # Pass the data to the template
    return render(request, 'test.html', {'data_json': data_json,'username': username})
def FileAudioForensic(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/FileAudioForensic")
    return render(request,'FileAudioForensic.html',{'username': username})
def testsample(request):
   
    return render(request,'testsample.html')
def LiveAudioForensic(request):
    username = request.user
    if request.user.is_anonymous:
        return redirect("/LiveAudioForensic")
    return render(request,'LiveAudioForensic.html',{'username': username})


def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        with open('media/' + uploaded_file.name, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        lemmatizer = WordNetLemmatizer()
        folder_path = "E:\Fyp Project\Crime Watch\Criminal"
        file_list = os.listdir(folder_path)
        audio_files = [os.path.join(folder_path, file) for file in file_list if file.endswith(('.mp3', '.wav', '.ogg','.m4a'))]

        audio_file= open('media/' + uploaded_file.name, "rb")
        transcript = openai.Audio.translate("whisper-1", audio_file)
        text_to_predict= transcript['text']


        # Make prediction and identify criminal words, locations, dates, and times
        
        
        
        prediction_result = predict_and_identify_criminal(text_to_predict)
        openai_api_key =  ""
        openai.api_key = openai_api_key

        # # Construct a prompt that includes the title, subtitle, and text
        prompt = f"Analyze the following sentence and provide extracted names, time, place, day, sentiment analysis (whether the sentiment is related to criminal or non-criminal activity), and whether the speaker is involved in the criminal activity or not:Sentence: {text_to_predict}"

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows the users instructions"},
                {"role": "user", "content": prompt}
            ]
        )
        message = completion.choices[0].message['content']
        print(message)

        # Print prediction and other extracted information
        context={
            "prediction":stringify_list_items(prediction_result["prediction"]),
            "Identifiedwords": stringify_list_items(prediction_result["criminal_words"]),
            "Summary":message ,
            "ProcessedText":text_to_predict
        }

        

        return JsonResponse(context)
def Lawyerresponse(request):
    if request.method == 'POST':
        openai_api_key =  ""
        openai.api_key = openai_api_key
        data = json.loads(request.body)
        message = data.get('message', '')
        print(message)
        
        # # Construct a prompt that includes the title, subtitle, and text
        prompt = f"Develop a conversational model that responds to questions related to legal matters, greetings, and punishment information punishments and law for various crimes under relevant legal systems of pakistan constitution if their in not mention in the question. If the input question falls outside these topics, the model should politely indicate its limitations and prompt the user to ask another question within those boundaries. The model should be able to provide information about punishments and law for various crimes under relevant legal systems of pakistan constitution and also artical number  if their in not mention in the question and detailed brife exlpain.My question is that {message }"

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            max_tokens=4000,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows the users instructions"},
                {"role": "user", "content": prompt}
            ]
        )
        message = completion.choices[0].message['content']
        
        context={
            "prediction":message
        }
        return JsonResponse(context)