#importing dependencies
from fastapi import FastAPI,File,UploadFile,HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import logging
import time

#The basicConfig configures the root logger.
logging.basicConfig(filename = 'test.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

#loading the deep learning model
model=load_model('/home/ubuntu/Downloads/malaria dataset/mal_model.h5')

#functcion which read and preprocess the image
def read_file_as_image(data) -> np.ndarray:
    #BytesIO are methods that manipulate string and bytes data in memory.
    image = Image.open(BytesIO(data))
    #resizing the image
    image=image.resize((224,224))
    #convert image into numpy array
    image=np.array(image)
    #expand the dimesnions
    image= np.expand_dims(image, 0)
    #rescaling the image
    image=image/255
    #returning the image
    return image


app = FastAPI()

@app.get('/')
def home():
    return {'welcome to malaria cell detection model'}


@app.post('/prediction')
async def predict(file: UploadFile = File(...)):
    # it will allow only jpg and jpeg,png format else it will through error
    if not file.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an right extension of image.')

    logging.info('Got the image from user')
    #task strating time
    start = time.perf_counter()
    
    try:
        #reading the image using read image function
        image = read_file_as_image(await file.read())
        logging.info(f'Successfully preprocessed {image}')

        #predict the image
        prediction = model.predict(image)
        logging.info(f'predictions {prediction}')

        # print('pred',predictions)
        if prediction==0:
            msg =  'Parasitized'
        else:
            msg = 'Uninfected'
    
        return {'prediction': msg}
        
    except Exception as E:
        logging.exception(f"Warning, the image preprocessing is not successful , Exception is {E}")
        logging.info("\n")
        return {}
    finally:
        # task ending time
        end=time.perf_counter()
        #total time that task has taken
        logging.info(f'Took {end-start}seconds for Prediction')
        logging.info('-'*100)
        logging.info('\n')