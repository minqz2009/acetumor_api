import torch
import torch.nn as nn
from torchvision import transforms, models
import json
from flask import jsonify

# Setting up the neural networks
crop_size = 224
resize_size = (336,448)
transform = transforms.Compose([transforms.Resize(resize_size),
                                transforms.RandomCrop(crop_size),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225]) # mean(r, g, b)  std(r, g, b)
                                ])

# Getting cuda if available, else use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading weights
BREAST_MODEL_PATH = "./models/densenet161(r(336, 448)c224b4)-90.pth"
COVID_MODEL_PATH = "./models/covid-densenet161(r(336, 448)c224b4)-96.pth"
BRAIN_MODEL_PATH = "./models/vgg16(r(336, 448)c224b8)-82.pth"

breast_net = models.densenet161(pretrained=False).to(device)
covid_net = models.densenet161(pretrained=False).to(device)
brain_net = models.vgg16(pretrained=False).to(device)

breast_net.load_state_dict(torch.load(BREAST_MODEL_PATH, map_location='cpu'))
covid_net.load_state_dict(torch.load(COVID_MODEL_PATH, map_location='cpu'))
brain_net.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location='cpu'))

# Map the result from the network to its label
breast_dict = ["Benign", "InSitu", "Invasive", "Normal"]
covid_dict = ["Covid", "Normal", "Viral"]
brain_dict = ["Glioma", "Meningioma", "Normal", "Pituitary"]

# Predicting results according to the image passed in and the position declared
def predict(position, image):
    # Detecting breast cancer
    if position == "Breast":
        breast_net.eval()
        with torch.no_grad():
            image = transform(image)
            output = breast_net(image.unsqueeze(0))

            prob = nn.functional.softmax(output[:4], dim=1) * 100
            _, predicted = torch.max(output, 1)

        distribution = {
            "normal":int(prob[0][3]), 
            "benign":int(prob[0][0]), 
            "insitu":int(prob[0][1]), 
            "invasive":int(prob[0][2])
        }
        reply = jsonify(
                            status="success",
                            result=breast_dict[predicted[0].item()],
                            distribution=distribution
                        )
        return reply, reply 
            
    # Detecting covid-19
    elif position == "Chest":
        covid_net.eval()
        with torch.no_grad():
            image = transform(image)
            print(image.shape)
            output = covid_net(image.unsqueeze(0))

            prob = nn.functional.softmax(output[:3], dim=1) * 100
            _, predicted = torch.max(output, 1)

        distribution = {
            "covid":int(prob[0][0]), 
            "normal":int(prob[0][1]), 
            "viral":int(prob[0][2])
        }
        reply = jsonify(
                            status="success",
                            result=covid_dict[predicted[0].item()],
                            distribution=distribution
                        )
        return reply, covid_dict[predicted[0].item()]

    # Detecting brain tumor
    elif position == "Brain":
        brain_net.eval()
        with torch.no_grad():
            image = transform(image)
            output = brain_net(image.unsqueeze(0))

            prob = nn.functional.softmax(output[:4], dim=1) * 100
            _, predicted = torch.max(output, 1)

        distribution = {
            "glioma":int(prob[0][0]), 
            "meningioma":int(prob[0][1]), 
            "normal":int(prob[0][2]),
            "pituitary":int(prob[0][3])
        }
        reply = jsonify(
                            status="success",
                            result=brain_dict[predicted[0].item()],
                            distribution=distribution
                        )
        return reply, brain_dict[predicted[0].item()]
