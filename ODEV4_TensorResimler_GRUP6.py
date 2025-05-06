import fasttext
import numpy as np
import os

# Model dosyasını yükle
model_file = "model_amzn.bin"
assert os.path.exists(model_file), f"{model_file} bulunamadı. Lütfen önce modeli eğitin."
model = fasttext.load_model(model_file)

# Ayar: Kaç kelime alınsın?
top_n = 750

# Tüm kelimeleri al ve en çok geçen ilk top_n tanesini filtrele
words = model.get_words()[:top_n]
vectors = [model.get_word_vector(word) for word in words]

# Vektörleri tensor.tsv olarak kaydet (her satırda bir kelime vektörü)
np.savetxt("tensor.tsv", vectors, delimiter="\t")

# Kelime etiketlerini metadata.tsv olarak kaydet (her satırda bir kelime)
with open("metadata.tsv", "w", encoding="utf-8") as f:
    for word in words:
        f.write(word + "\n")

print("✅ 'tensor.tsv' ve 'metadata.tsv' dosyaları oluşturuldu. Şimdi bunları https://projector.tensorflow.org adresine yükleyebilirsin.")
