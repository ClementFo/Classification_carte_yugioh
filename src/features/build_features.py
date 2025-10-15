import pandas as pd
import numpy as np
import cv2
from PIL import Image

import os


def build_mvtec_dataframe() -> pd.DataFrame:
    """
    Génère un dataframe à partir d'un CSV

    Return:
        Dataframe des varialbles les plus intéressantes
    """
    
    df = pd.read_csv('data/Yugi_db_cleaned.csv', sep=";")
    df = df.drop(["Card-set", "Card_number", "Rarity", "ATK / LINK",
                  "Other names (Japanese)", "Summoned by the effect of", "Password",
                  "Status", "Other names", "Attribute", "Ritual required",
                  "Ritual Monster required", "Effect types", "Property"], axis=1)
    
    df["Card classification"] = df.apply(
        lambda row: 1 if row["Card type"] == "Monster" and "Ritual" in row["Types"]
        else 2 if row["Card type"] == "Monster" and "Fusion" in row["Types"]
        else 3 if row["Card type"] == "Monster" and "Synchro" in row["Types"]
        else 4 if row["Card type"] == "Monster" and "Xyz" in row["Types"]
        else 5 if row["Card type"] == "Monster" and "Link" in row["Types"]
        else 6 if row["Card type"] == "Monster" and "Pendulum" in row["Types"]
        else 0 if row["Card type"] == "Monster"
        else 7 if row["Card type"] == "Spell"
        else 8,
        axis=1)
    df["Types"] = df.apply(
        lambda row: row["Types"] if row["Types"] is not None and row["Types"] != ""
        else "Spell" if row["Card type"] == "Spell"
        else "Trap",
        axis=1)
    
    df["Level"] = df["Level"].apply(lambda x: 1 if x is not None else 0)
    df["ATK / DEF"] = df["ATK / DEF"].apply(lambda x: 1 if x is not None else 0)
    df["Pendulum Scale"] = df["Pendulum Scale"].apply(
        lambda x: 1 if x is not None else 0)
    df["Type Level"] = df.apply(lambda row: 0 if row["Level"] is not None
                                else 1 if row["Rank"] is not None
                                else 2 if row["Link Arrows"] is not None
                                else 3,
                                axis=1)
    
    df = df.drop(["Level", "Rank", "Link Arrows"], axis=1)

    return df


def gaussian_blur(images, size=(3, 3),
                  type_threshold=cv2.THRESH_BINARY,
                  sigmaX=0,
                  to_gray=False,
                  flatten=True):
    """
    Applique le filtre Gaussian Blur aux images.

    Parameters:
        images: les images à traiter
        size: taille du kernel du filtre gaussian
        type_threshold: type de seuillage à appliquer
         sigmaX: écart-type du filtre
        to_gray: permet de faire passer une image en couleur en noire et blanc
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    seuils = []
    img_seuillages = []
    for image in images:
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtre = cv2.GaussianBlur(image, ksize=size, sigmaX=sigmaX)
        seuil, img_seuillage = cv2.threshold(filtre,
                                             120,
                                             255,
                                             type=type_threshold)
        seuils.append(seuil)
        if flatten:
            img_seuillages.append(img_seuillage.ravel().astype(np.float32))
        else:
            img_seuillages.append(img_seuillage)

    return seuils, img_seuillages


def gaussian_blur_canny(images,
                        size=(3, 3),
                        sigmaX=0,
                        t_lower=1,
                        t_upper=200,
                        to_gray=False,
                        flatten=True):
    """
    Applique le filtre Canny aux images.

    Parameters:
        images: les images à traiter
        size: taille du kernel du filtre gaussian
        cvtColor: permet de faire passer une image en couleur en noire et blanc
        sigmaX: écart-type du filtre
        t_lower, t_upper: valeur de seuil du seuillage
        to_gray: permet de faire passer une image en couleur en noire et blanc
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    all_edges = []
    for image in images:
        if to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        filtre = cv2.GaussianBlur(image, ksize=size, sigmaX=sigmaX)

        edges = cv2.Canny(filtre, t_lower, t_upper)
        if flatten:
            edges = edges.flatten()

        all_edges.append(edges)
    return all_edges


def gaussian_blur_canny_one_image(image,
                                  size=(3, 3),
                                  sigma=0,
                                  t_lower=1,
                                  t_upper=200,
                                  to_gray=False):
    """
    Applique le filtre Canny a une image.

    Parameters:
        images: l'image à traiter
        size: taille du kernel du filtre gaussian
        cvtColor: permet de faire passer une image en couleur en noire et blanc
        sigmaX: écart-type du filtre
        t_lower, t_upper: valeur de seuil du seuillage
        to_gray: permet de faire passer une image en couleur en noire et blanc
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une d'image filtrées
    """
    if to_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtre = cv2.GaussianBlur(image, ksize=size, sigmaX=sigma)

    edges = cv2.Canny(filtre, t_lower, t_upper)
    return edges


def houghLinesP(images, rho=3, theta=np.pi / 20,
                threshold=100, minLineLength=0,
                maxLineGap=20, to_gray=False, flatten=True):

    """
    Filtre appliquant des lignes sur un ensemble d'images via leurs contours
    généré par le filtre Canny

    Parameters:
        images : liste des images à analyser
        rho : La résolution du paramètre r en pixels.
        theta : La résolution du paramètre theta en radians.
        threshold: Le nombre minimum d'intersections pour "détecter" une ligne
        minLinLength: Le nombre minimum de points qui peuvent former une ligne.
        maxLineGap: Ecart maximum entre deux points dans une même ligne.
        to_gray: permet de faire passer une image en couleur en noire et blanc
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    all_lines = []
    for image in images:
        edges = gaussian_blur_canny_one_image(image, to_gray=to_gray)
        lines = cv2.HoughLinesP(
            edges,
            rho=rho,
            theta=theta,
            threshold=threshold,
            minLineLength=minLineLength,
            maxLineGap=maxLineGap
        )
        line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1),
                             (x2, y2),
                             color=[255, 0, 0],
                             thickness=3)

        piece_lines = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                                      0.8,
                                      line_img,
                                      1.0,
                                      0.0)
        if flatten:
            piece_lines = piece_lines.flatten()
        all_lines.append(piece_lines)
    return all_lines


def sobel(images, ddepth=cv2.CV_64F, dx=1, dy=0, flatten=True):
    """
    Filtre Sobel pour une liste d'images

    Parameters:
        images : liste des images à analyser
        ddepth : profondeur de l'image de sortie
        dx et dy : détermine la mise en évidence des bords verticaux
                    ou horizontaux de l'image
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    sobels = []
    for img in images:
        sobel = cv2.Sobel(img, ddepth=ddepth, dx=dx, dy=dy)
        if flatten:
            sobel = sobel.flatten()
        sobels.append(sobel)
    return sobels


def laplacian(images, ddepth=cv2.CV_64F, flatten=True):
    """
    Filtre Laplacien pour une liste d'images

    Parameters:
        images : liste des images à analyser
        ddepth : profondeur de l'image de destination
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    laplacians = []
    for img in images:
        laplacian = cv2.Laplacian(img, ddepth=ddepth)
        if flatten:
            laplacian = laplacian.flatten()
        laplacians.append(laplacian)
    return laplacians


def erosion(images, size=(3, 3), flatten=True):
    """
    Erosion pour une liste d'images

    Parameters:
        images : liste des images à analyser
        size : niveau d'érosion de l'image
        flatten: si True, retourne un vecteur 1D pour chaque image
    Return:
        une liste d'images filtrées ou de vecteurs 1D si flatten=True
    """
    img_erosions = []
    kernel_1 = np.ones(size, np.uint8)

    for img in images:
        erosion = cv2.erode(img, kernel_1)
        if flatten:
            erosion = erosion.flatten()
        img_erosions.append(erosion)

    return img_erosions


def load_images_from_df(df, folder, resize=None, to_gray=False):
    images, labels = [], []
    for _, row in df.iterrows():
        img_path = folder + "/" + row["Image_name"]
        try:
            img = Image.open(img_path)
            # Supprime le profil ICC pour éviter le warning
            if "icc_profile" in img.info:
                img.info.pop("icc_profile")
            if to_gray:
                img = img.convert("L")  # Convertir en grayscale
            else:
                img = img.convert("RGB")
            if resize is not None:
                img = img.resize(resize)
            # Convertir en numpy array si besoin
            img_array = np.array(img)
            images.append(img_array)
            labels.append(row["Card classification"])
        except Exception as e:
            print(f"Erreur lecture image {img_path}: {e}")
    return images, labels


def resize_with_aspect_ratio(img, target_height, target_width):
    h, w = img.shape[:2]
    # Calcul du ratio de redimensionnement
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Redimensionnement
    resized_img = cv2.resize(img, (new_w, new_h))

    # Déterminer le nombre de canaux
    if len(resized_img.shape) == 3:
        channels = resized_img.shape[2]
    else:
        channels = 1

    # Création d'une image vide avec la taille cible
    if channels == 1:
        canvas = np.zeros((target_height, target_width), dtype=resized_img.dtype)
    else:
        canvas = np.zeros((target_height, target_width, channels),
                          dtype=resized_img.dtype)

    # Centrer l'image redimensionnée
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
    return canvas


def process_images(input_dir, output_base_dir, list_func):
    """
    Parcourt input_dir, applique la liste de fonctions à chaque images,
    et sauvegarde le résultat dans output_base_dir/func_name/
    en respectant la même hiérarchie.

    Parameters:
    - input_dir : Chemin vers le dossier source
    - output_base_dir : Chemin vers le dossier de sortie
    - list_func : Liste de fonction de transformation (prend une image OpenCV
                    et retourne une image)
    """
    for root, dirs, files in os.walk(input_dir):
        print(root)

        # Filtrer uniquement les fichiers image
        img_files = [f for f in files if f.lower().endswith((".png",
                                                             ".jpg",
                                                             ".jpeg",
                                                             ".bmp",
                                                             ".tiff"))]
        if not img_files:
            continue

        # Charger toutes les images du dossier
        imgs = []
        paths = []
        for file in img_files:
            input_path = os.path.join(root, file)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Impossible de lire {input_path}, ignoré.")
                continue
            imgs.append(img)
            paths.append(file)

        if not imgs:
            continue

        # Appliquer chaque fonction sur TOUTES les images du dossier
        for func_name, func in list_func.items():
            processed_imgs = func(imgs)

            # Si une seule image est renvoyée, on l'enveloppe en liste
            if isinstance(processed_imgs, np.ndarray):
                processed_imgs = [processed_imgs]
            elif not isinstance(processed_imgs, (list, tuple)):
                print(
                    f"La fonction {func_name} a renvoyé un type inattendu "
                    f"({type(processed_imgs)})."
                )
                continue

            # Vérifier correspondance nb images
            if len(processed_imgs) != len(paths):
                print(
                    f"La fonction {func_name} n’a pas renvoyé le bon nombre d’images "
                    f"({len(processed_imgs)} au lieu de {len(paths)})."
                )
                continue

            # Sauvegarder chaque image
            for proc_img, file in zip(processed_imgs, paths):
                if proc_img.dtype != np.uint8:
                    if np.issubdtype(proc_img.dtype, np.floating):
                        proc_img = np.clip(proc_img, 0, 1) * 255
                    proc_img = proc_img.astype(np.uint8)

                relative_path = os.path.relpath(root, input_dir)
                output_dir = os.path.join(output_base_dir, func_name, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, proc_img)
                print(f"Sauvegardé: {output_path}")


def process_filtre():
    filtres = {
        "Gaussian Blur": lambda X: gaussian_blur(X, flatten=False)[1],
        "Canny": lambda X: gaussian_blur_canny(X, flatten=False),
        "HoughLinesP": lambda X: houghLinesP(X, flatten=False),
        "Sobel": lambda X: sobel(X, flatten=False),
        "Laplacian": lambda X: laplacian(X, flatten=False),
        "Erosion": lambda X: erosion(X, flatten=False),
    }

    process_images(
        input_dir="data/Yugi_images",
        output_base_dir="data/Yugi_images_processed",
        list_func=filtres
    )
