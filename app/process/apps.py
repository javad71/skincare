import os
import torch
from django.apps import AppConfig
from .core.acne.fastrcnn_modules import create_model
from deepface import DeepFace


class ProcessConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'process'
    def ready(self):
        if os.environ.get('RUN_MAIN'):
            # ----------------------------------------------------------------------------------
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print('device: ', device)
            global acne_model, age_model, race_model
            acne_model = create_model(num_classes=7).to(device)
            acne_model.load_state_dict(torch.load('./process/core/acne/model48.pth', map_location=device))
            acne_model.eval()
            print("Acne model loaded")
            # ----------------------------------------------------------------------------------
            age_model = DeepFace.build_model('Age')
            print("Age model loaded")
            race_model = DeepFace.build_model('Race')
            print("Race model loaded")
