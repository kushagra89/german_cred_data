import pandas as pd
import pickle
PREPROCESSOR_PATH = 'model/preprocessor.pkl'
MODEL_PATH = 'model/model.pkl'

def predict(input_csv):
    df = pd.read_csv(input_csv)

    with open(PREPROCESSOR_PATH,'rb') as f:
        preproc = pickle.load(f)

    with open(MODEL_PATH,'rb') as f:
        model = pickle.load(f)
    
    x = preproc.transform(df)

    preds = model.predict(x)

    df['prediction'] = preds
    df.to_csv('predictions.csv',index = False)
    print('predictions saved to predictions.csv')

if __name__=="__main__":
    predict("data/new/new_data.csv")