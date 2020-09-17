# AceTumor API
Currently support the detection of **brain tumor**, **breast tumor** and **pneumonia** (**COVID-19** and others).

**Densenet-161** is used in predicting breast tumor and pneumonia. 

**VGG-16** is used in predicting brain tumor.

To run the API:

    python3 run.py

<br>

# API Formats
## API Request Format (POST, JSON)
    {
        position: (one of: "Brain", "Breast", "Chest"),
        img: (raw image byte encoded with base64 and utf-8)
    }

### Example
```python
import requests
import base64
img = open('PATH TO IMAGE FILE', 'rb').read()
img_str = base64.b64encode(img)
img_str = img_str.decode("utf-8")
data = {'position':"Chest", # "Breast", "Brain"
        'img': img_str}
r = requests.post('URL/api/upload-img', json=data)
```
<br>

## API Response Format (JSON) 
    {
        status: (response status),
        result: (label of the classification),
        distribution: {
            (see details below)
        }
    }

Note that for the "distribution" field, the result varies according to the "position" field in the request. If the position is:

**Brain:**

    {
        glioma: (probability),
        meningioma: (probability),
        normal: (probability),
        pituitary: (probability)
    }

<br>

**Breast:**

    {
        normal: (probability),
        benign: (probability),
        insitu: (probability),
        invasive: (probability)
    }

<br>

**Chest:**

    {
        covid: (probability),
        normal: (probability),
        viral: (probability),
    }

<br>

# Requirements

See the **requirements.txt**