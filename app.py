import preprocess
import train_model
import mlflow_tracking

def main():
    print("Veri işleme başlatılıyor...")
    preprocess.preprocess_data('data/opinions.csv')
    
    print("Model eğitimi başlatılıyor...")
    train_model.train()
    
    print("MLflow izleme başlatılıyor...")
    mlflow_tracking.mlflow_example()

if __name__ == "__main__":
    main()
