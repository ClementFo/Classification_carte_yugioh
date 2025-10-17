import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from django.conf import settings
from django.http import HttpResponse

import src.features.build_features as bf

matplotlib.use('Agg')


def project_path(*parts):
    return os.path.join(settings.BASE_DIR, *parts)


def visualisation(request):
    df = bf.build_mvtec_dataframe()

    visu_b = visual_base(df)
    visu_f = visual_filtre(df)

    html_http = '<h1>Visualisation</h1>'
    html_http += '<h2>Visualisation général des données</h2>'
    html_http += '<p>Ce premier graphique le nombre de carte par catégorie de carte</p>'
    html_http += f"<img src='{settings.MEDIA_URL}{os.path.basename(visu_b)}'/>"
    html_http += '<h2>Visualisation image avec filtre</h2>'
    i = 0
    for visu in visu_f:
        carte = "monstre" if i == 0 else "piège" if i == 1 else "magie"
        html_http += f'<p>Cette série de carte montre différents filtres pour des \
              cartes {carte}</p>'
        html_http += f"<img src='{settings.MEDIA_URL}{os.path.basename(visu)}'/>"
        i += 1
    return HttpResponse(html_http)


def visual_base(df):

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

    # ---- Titre global ----
    fig.suptitle('Catégorie', fontsize=14)

    out_dir = project_path('reports', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    name_file = os.path.join(out_dir, "Visualisation1.png")
    plt.tight_layout()
    plt.savefig(name_file)
    plt.close(fig)
    return name_file


def visual_filtre(df):
    # Image filtres
    types = df['Card type'].unique()
    names_files = []
    key = 0

    categories = df['Card classification'].unique()
    occurence = [df.loc[
        df['Card classification'] == cat].count().iloc[0] for cat in categories]
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
        img_path = "data/Yugi_images/processed/Canny/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre Canny")
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 3, 2)
        img_path = "data/Yugi_images/processed/Erosion/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre Erosion")
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 3, 3)
        img_path = "data/Yugi_images/processed/Gaussian Blur/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre Gaussian Blur")
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 3, 4)
        img_path = "data/Yugi_images/processed/HoughLinesP/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre HoughLinesP")
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 3, 5)
        img_path = "data/Yugi_images/processed/Laplacian/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre Laplacian")
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(2, 3, 6)
        img_path = "data/Yugi_images/processed/Sobel/" + image_name
        img = cv2.imread(img_path)
        plt.imshow(img, cmap='gray')
        plt.title("Filtre Sobel")
        plt.xticks([])
        plt.yticks([])

        out_dir = project_path('reports', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        name_file = os.path.join(out_dir, f"Filtre_type_{key}.png")
        names_files.append(name_file)
        plt.savefig(name_file)
        key += 1
        plt.close(fig)
    return names_files
