import cv2

import matplotlib.pyplot as plt
import seaborn as sns

import src.features.build_features as bf

df = bf.build_mvtec_dataframe()

rename_dict = {
    0: 'Monstre',
    1: 'Ritual',
    2: 'Fusion',
    3: 'Synchro',
    4: 'Xyz',
    5: 'Link',
    6: 'Pendulum',
    7: 'Spell',
    8: 'Trap'
}
df['Card_classification_name'] = df['Card classification'].replace(rename_dict)
# df.to_csv('data/Yugi_dataframe.csv', index=False)

fig, axes = plt.subplots(1, 1, figsize=(12, 5))

# ---- Histogramme (à gauche) ----
sns.countplot(x='Card_classification_name', data=df)

# ---- Calcul des occurrences pour le camembert ----
categories = df['Card classification'].unique()
occurence = [df.loc[
    df['Card classification'] == cat].count().iloc[0] for cat in categories]

# ---- Titre global ----
fig.suptitle('Catégorie', fontsize=14)

plt.tight_layout()
plt.savefig("reports/figures/Visualisation1.png")
plt.show()


# Image filtres
types = df['Card type'].unique()
key = 0
for onetype in types:
    occurence = df.loc[df['Card type'] == onetype]
    if occurence.empty:
        print(f"Aucune image trouvée pour le type {onetype}")
        continue
    
    image_name = occurence['Image_name'].iloc[0]
    img_path = "data/Yugi_images/" + image_name
    img = cv2.imread(img_path)
    if img is None:
        print("Impossible de lire l'image")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Image normale {onetype}")
    plt.xticks([])
    plt.yticks([])

    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(2, 3, 1)
    img_path = "data/Yugi_images_processed/Canny/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre Canny")
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(2, 3, 2)
    img_path = "data/Yugi_images_processed/Erosion/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre Erosion")
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(2, 3, 3)
    img_path = "data/Yugi_images_processed/Gaussian Blur/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre Gaussian Blur")
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(2, 3, 4)
    img_path = "data/Yugi_images_processed/HoughLinesP/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre HoughLinesP")
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(2, 3, 5)
    img_path = "data/Yugi_images_processed/Laplacian/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre Laplacian")
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(2, 3, 6)
    img_path = "data/Yugi_images_processed/Sobel/" + image_name
    img = cv2.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.title("Filtre Sobel")
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f"reports/figures/Filtre_type_{key}.png")
    key += 1
    plt.show()
    
