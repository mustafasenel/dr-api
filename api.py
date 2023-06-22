import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from keras.models import load_model
model = load_model("model-dr.h5")


def preprocess(img):
    cv2_img = np.array(img)
    image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(400,400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kopya = image.copy()
    kopya = cv2.cvtColor(kopya, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(kopya,(5,5),0)
    thresh = cv2.threshold(blur,10,255, cv2.THRESH_BINARY)[1]
    kontur = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontur = kontur[0][0]
    kontur = kontur[:,0,:]
    x1 = tuple(kontur[kontur[:,0].argmin()])[0]
    y1 = tuple(kontur[kontur[:,1].argmin()])[1]
    x2 = tuple(kontur[kontur[:,0].argmax()])[0]
    y2 = tuple(kontur[kontur[:,1].argmax()])[1]
    x = int(x2-x1)*4//50
    y = int(y2-y1)*5//50
    kopya2 = image.copy()
    if x2-x1 >100 and y2-y1> 100:
        kopya2 = kopya2[y1+y : y2-y , x1+x : x2-x]
        kopya2 = cv2.resize(kopya2,(400,400))
    lab = cv2.cvtColor(kopya2, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0,tileGridSize=((8,8)))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    son = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    med_son = cv2.medianBlur(son, 3)
    arka_plan = cv2.medianBlur(son, 37)
    maske = cv2.addWeighted(med_son,1,arka_plan,-1,255)
    son_img = cv2.bitwise_and(maske,med_son)
    return son_img

app = FastAPI()

@app.get('/')
async def index():
  
    return "Welcome DR Prediction"

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())

    image = preprocess(image)

    prediction = model.predict(np.expand_dims(image,axis=0))[0]

    no_dr = format(prediction[0]*100,".2f")
    mild = format(prediction[1]*100,".2f")
    modere = format(prediction[2]*100,".2f")
    severe = format(prediction[3]*100,".2f")
    pdr = format(prediction[4]*100,".2f")

    return {"no_dr":no_dr,
            "mild":mild,
            "modere":modere,
            "severe":severe,
            "pdr":pdr,
            }


