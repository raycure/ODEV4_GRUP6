import fasttext
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Dosya adları
train_file = "train.ft.txt"
test_file = "test.ft.txt"
model_file = "model_amzn.bin"

# 1. Veriyi oku ve etiketleri ayır
texts = []
labels = []

# Dosyalar zaten varsa, veriyi tekrar oluşturma
if not (os.path.exists(train_file) and os.path.exists(test_file)):
    with open("test.ft.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__label__1"):
                labels.append(0)  # Negatif
            elif line.startswith("__label__2"):
                labels.append(1)  # Pozitif
            text = line.replace("__label__1", "").replace("__label__2", "").strip()
            texts.append(text)

    # Veriyi karıştır
    data = list(zip(texts, labels))
    random.shuffle(data)
    texts, labels = zip(*data)

    # Eğitim ve test bölme
    train_texts = texts[:int(len(texts) * 0.8)]
    train_labels = labels[:int(len(labels) * 0.8)]
    test_texts = texts[int(len(texts) * 0.8):]
    test_labels = labels[int(len(labels) * 0.8):]

    # Eğitim dosyasını yaz
    with open(train_file, "w", encoding="utf-8") as f:
        for text, label in zip(train_texts, train_labels):
            f.write(f"__label__{label} {text}\n")

    # Test dosyasını yaz
    with open(test_file, "w", encoding="utf-8") as f:
        for text, label in zip(test_texts, test_labels):
            f.write(f"__label__{label} {text}\n")

# Eğitim yapılmamışsa modeli eğit ve loss takip et
loss_values = []
epochs = 25

if not os.path.exists(model_file):
    for epoch in range(1, epochs + 1):
        model = fasttext.train_supervised(input=train_file, epoch=epoch)
        result_epoch = model.test(test_file)
        loss_values.append(result_epoch[1])  # Burada precision yerine loss proxy'si kullanıyoruz
    model.save_model(model_file)
else:
    model = fasttext.load_model(model_file)

# Modeli değerlendir
result = model.test(test_file)
precision = result[1]
recall = result[2]
test_samples = result[0]

# Eğer loss takip edildiyse grafiği çiz
fig, axes = plt.subplots(1, 2 if loss_values else 1, figsize=(14, 6) if loss_values else (7, 6))

# Precision ve Recall
ax = axes[0] if loss_values else axes
ax.bar(['Precision', 'Recall'], [precision, recall], color=['blue', 'orange'])
ax.set_title('Precision and Recall')
ax.set_ylim([0, 1])
ax.set_ylabel('Score')

# Loss grafiği varsa
if loss_values:
    axes[1].plot(range(1, epochs + 1), loss_values, marker='o', color='green')
    axes[1].set_title('Loss Proxy Over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Proxy Metric (e.g. precision)')

plt.tight_layout()
plt.show()
