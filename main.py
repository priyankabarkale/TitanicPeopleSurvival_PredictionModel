from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_building
from src.model_evaluation import model_evaluation

def main():
    df=data_ingestion()
    X_train,X_test,y_train,y_test=data_preprocessing(df)
    model=model_building(df)
    classification_report,confusion_matrix=model_evaluation(model,X_train,X_test,y_train,y_test)
    print(confusion_matrix)
    print(classification_report)
    
    
    


if __name__ == "__main__":
    main()
