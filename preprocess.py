import pandas as pd
import re

# Veri temizleme fonksiyonu
def clean_text(text):
    # Özel karakterleri ve sayıları kaldır
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Küçük harfe çevir
    text = text.lower()
    # Fazla boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Veri işleme fonksiyonu
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Text temizliği
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Temizlenmiş veriyi kaydetme
    df.to_csv('data/processed_opinions.csv', index=False)
    return df

if __name__ == "__main__":
    filepath = 'data/opinions.csv'  # İşlenecek veri seti
    preprocess_data(filepath)
    print("Veri temizleme işlemi tamamlandı.")
