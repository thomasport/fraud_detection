# Fraud Detection

Machine learning model to detect fraudulent transacions. Based on the kaggle dataset [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 



    ├───analysis  Notebook with data analysis and modeling
    ├───app       Application to serve model prediction
    ├───assets    Binaries and data files
    └───src       Scripts to support analysis
## Installing
Run the command to create environment and install all the required packages

    poetry install 


## Run API 
Inside app folder, run:      

    poetry run fastapi dev app.py 