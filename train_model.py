import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Veriyi yükleme ve model eğitimi
def train():
    # Veriyi yükleme
    data = pd.read_csv('data/processed_opinions.csv')
    
    # Öznitelik ve etiket ayırma
    X = data['cleaned_text']  # Giriş verisi
    y = data['label']  # Sınıf etiketleri (örneğin claim, counterclaim vs.)
    
    # Eğitim ve test verisi ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Basit bir RandomForest modelini eğitme
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Tahmin yapma
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Modelin doğruluğunu yazdırma
    print(f"Model Doğruluğu: {accuracy * 100:.2f}%")
    
    # MLflow ile izleme
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "models/random_forest")
    
if __name__ == "__main__":
    mlflow.start_run()  # MLflow izlemesini başlat
    train()
    mlflow.end_run()    # MLflow izlemesini bitir
