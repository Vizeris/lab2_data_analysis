import warnings
warnings.filterwarnings('ignore')
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

url = "https://ekipland.ru/reviews/wp-content/uploads/2022/07/filip-5i3xkbd9njm-unsplash.jpg"

response = requests.get(url)
img = Image.open(BytesIO(response.content))

image = np.array(img, dtype=np.float64) / 255

w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))

unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
print(f"Original image has {unique_colors.shape[0]} unique colors.")

def quantize_image(n_colors):
    print(f"Fitting model for {n_colors} colors...")
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    return recreate_image(kmeans.cluster_centers_, labels, w, h)

def recreate_image(codebook, labels, w, h):
    return codebook[labels].reshape(w, h, -1)

plt.figure(1, figsize=(10, 10))
plt.clf()
plt.axis("off")
plt.title("Original image")
plt.imshow(image)

for i, n_colors in enumerate([64, 32, 16, 8], 2):
    quantized_image = quantize_image(n_colors)
    plt.figure(i, figsize=(10, 10))
    plt.clf()
    plt.axis("off")
    plt.title(f"Quantized image ({n_colors} colors)")
    plt.imshow(quantized_image)

plt.show()


#####################################################



import pandas as pd
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# spam.csv
# /kaggle/input/spam-emails/spam.csv
df = pd.read_csv('spam.csv', encoding='latin-1')
df.columns = ['Category', 'Message']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Processed_Message'] = df['Message'].apply(preprocess_text)

print(df.head())

def visualize(label):
    text = ' '.join(df[df['Category'] == label]['Processed_Message'])
    wordcloud = WordCloud(width=600, height=400, background_color="white").generate(text)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')

X_train, X_test, y_train, y_test = train_test_split(df['Processed_Message'], df['Category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))


################################################################################


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# proc_heart_cleve_3_withheader
# /kaggle/input/heart-disease-dataset/proc_heart_cleve_3_withheader

df = load_iris()
X = pd.DataFrame(df.data, columns=df.feature_names)
y = df.target

# df = pd.read_csv('proc_heart_cleve_3_withheader.csv')
X = df.drop('Disease', axis=1)
y = df['Disease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.tight_layout()
plt.show()

