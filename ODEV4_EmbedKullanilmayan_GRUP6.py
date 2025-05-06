import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss, precision_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
# 1. Veriyi oku ve etiketleri ayır
texts = []
labels = []

with open("test.ft.txt", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__label__1"):
            labels.append(0)
        elif line.startswith("__label__2"):
            labels.append(1)
        else:
            continue
        text = line.replace("__label__1", "").replace("__label__2", "").strip()
        texts.append(text)

# 2. Textleri dönüştür
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# 3. Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli tanımla
model = LogisticRegression(max_iter=1, warm_start=True, solver="lbfgs")
epochs = 30
losses = []

for epoch in range(epochs):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_train)
    loss = log_loss(y_train, y_proba)
    losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

# 5. Test verisiyle tahmin yap
y_pred = model.predict(X_test)

# 6. Metrikleri hesapla
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {acc:.4f}")
# a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"])
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()
# 7. Loss grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), losses, marker="o", color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Süresince Loss Grafiği")
plt.grid(True)

# 8. Metrikler bar grafiği
plt.subplot(1, 2, 2)
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [acc, prec, rec, f1]
colors = ['skyblue', 'orange', 'lightgreen', 'violet']
plt.bar(metrics, values, color=colors)
plt.ylim(0, 1)
plt.title("Model Performansı (Test Verisi)")
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
