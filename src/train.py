import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROCESSES_DATA_PATH = 'processed/processed_data.pkl'
MODEL_PATH = 'model/model.pkl'
METRICS_PATH = 'model/metrics.json'

def main():
    with open(PROCESSES_DATA_PATH,'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    #train model
    model = LogisticRegression(max_iter = 500)
    model.fit(X_train, y_train)
    #evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds, output_dict= True)
    cm = confusion_matrix(y_test, preds).tolist()

    #save model
    with open(MODEL_PATH,'wb') as f:
        pickle.dump(model, f)

    with open(METRICS_PATH,'w') as f:
        json.dump({"accuracy":acc, "classification_report":report,"confusion_matrix":cm},f,indent=4)
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"accuracy:{acc:.4f}")

if __name__ == "__main__":
    main()