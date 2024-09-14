import os
import face_recognition
import joblib

faces_folder = 'models'

face_encodings = []
face_names = []

for person_folder in os.listdir(faces_folder):
    person_path = os.path.join(faces_folder, person_folder)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            if filename.endswith('.jpg'):
                image = face_recognition.load_image_file(os.path.join(person_path, filename))
                face_encoding = face_recognition.face_encodings(image)
                if len(face_encoding) > 0:
                    face_encodings.append(face_encoding[0])
                    face_names.append(person_folder)

joblib.dump(face_encodings, 'face_encodings.pkl')
joblib.dump(face_names, 'face_names.pkl')
