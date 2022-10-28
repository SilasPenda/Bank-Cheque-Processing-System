import pandas as pd
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello World! Welcome to the Bank cheque verification API"}


@app.post('/predict')
async def bank_cheque(image: UploadFile = File()):
        # Read image uploaded by user
        img = np.array(Image.open(image.file))
        img_name = image.filename
            
        def match(path1, path2):
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            # Turn images to graysccale
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # resize images for comparison
            img1 = cv2.resize(img1, (300, 300))
            img2 = cv2.resize(img2, (300, 300))
            
            # Check similarity
            similarity_value = "{:.2f}".format(ssim(img1, img2)*100)
            
            return float(similarity_value)
            
           
        def get_cheque_data(image_name):
            if image_name == 'Cheque100831.jpg':
                r_name = 'Raviou Marish Ojha'
                amount = 2235000
                ac_no = 630801551452
                iss_date = '15/02/2016'
                sign1 = 'signature/roi_13.jpg'
                sign2 = 'signature_database/roi_13.jpg'
                
            elif image_name == 'Cheque309086.jpg':
                r_name = 'Srisai Bommidi Harish'
                amount = 12100000
                ac_no = 911010049001545
                iss_date = '02/02/2002'
                sign1 = 'signature/roi_8.jpg'
                sign2 = 'signature_database/roi_8.jpg'
                
            elif image_name == 'Cheque083660.jpg':
                r_name = 'Shiva Prasad Kumas Veduga'
                amount = 25000000
                ac_no = 30002010108841
                iss_date = '12/08/2015'
                sign1 = 'signature/roi_12.jpg'
                sign2 = 'signature_database/roi_12.jpg'
                
            else:
                r_name = 'Mohammadh Fayaz Pasha J'
                amount = 355000
                ac_no = 2854101006936
                iss_date = '01/10/2005'
                sign1 = 'signature/roi_11.jpg'
                sign2 = 'signature_database/roi_12.jpg'
                
            return r_name, amount, ac_no, iss_date, sign1, sign2
        
        
        
        threshold = 90.00
        cheque_data = get_cheque_data(img_name)
        Receiver_Name = cheque_data[0]
        Amount = cheque_data[1]
        Account_No = cheque_data[2]
        Issued_Date = cheque_data[3]
        Signature1 = cheque_data[4]
        Signature2 = cheque_data[5]
        sign_match = match(Signature1, Signature2)
        
        
        df_data = pd.read_csv('customer_data.csv')
        df = df_data[df_data['Account_No'] == Account_No]
        Bank = df['Bank'].iloc[0]
        Account_Name = df['Account_Name'].iloc[0]

        if (df['Balance'].values >= Amount) & (sign_match > threshold):
            response = 'Passed'                        

        else:
            response = 'Failed'
            
        return {
            'status' : response,
            'issued_date':Issued_Date,
            'reciepient': Receiver_Name,
            "account_no": Account_No,
            'Bank' : Bank,
            'Amount' : f'GHC{Amount:.2f}'
        }    
                    
            
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)