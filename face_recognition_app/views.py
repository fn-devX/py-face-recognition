import os
import joblib
import face_recognition
from PIL import Image
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

known_faces = joblib.load(os.path.join(BASE_DIR, 'face_encodings.pkl'))
known_face_names = joblib.load(os.path.join(BASE_DIR, 'face_names.pkl'))

def home(request):
    return render(request, 'home.html')

def recognize_face(request):
    if request.method == 'POST' and request.FILES['image']:
        image = Image.open(request.FILES['image'])

        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            return JsonResponse({'name': name})

    return JsonResponse({'name': 'Unknown'})