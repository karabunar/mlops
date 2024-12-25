import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# MLflow Tracking için örnek
def mlflow_example():
    # Örnek veri seti
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Model eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # MLflow ile izleme başlatma
    mlflow.start_run()
    
    # Hiperparametreleri kaydetme
    mlflow.log_param("n_estimators", 100)
    
    # Modelin doğruluğu
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Modeli kaydetme
    mlflow.sklearn.log_model(model, "models/random_forest")
    
    # Run işlemini bitir
    mlflow.end_run()
    
if __name__ == "__main__":
    mlflow_example()
    print("MLflow izleme işlemi tamamlandı.")
