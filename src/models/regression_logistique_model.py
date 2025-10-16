import os
import argparse
import logging
from typing import Any, Dict, cast

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

import src.features.build_features as bf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", ".*iCCP.*")
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


# Analyse des arguments en ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', help='Entrainement')
parser.add_argument('-e', '--eval', action='store_true', help='Evaluation')
args = parser.parse_args()

filtres = ["/", "/processed/Canny/",
           "/processed/Erosion/", "/processed/Gaussian Blur/",
           "/processed/HoughLinesP/", "/processed/Laplacian/",
           "/processed/Sobel/"]

df = bf.build_mvtec_dataframe()
df_results = pd.DataFrame()

max_iter = 10000
C = 1
parameter = [["l1", "liblinear", None],
             ["l2", "lbfgs", None], ["l2", "liblinear", None],
             ["l2", "newton-cg", None]]
model_type = "regression_logistique"

X_tmp, y_tmp = bf.load_images_from_df(df, filtres)
target_height = 256
target_width = 177
X_tmp_resized = [
    bf.resize_with_aspect_ratio(img, target_height, target_width) for img in X_tmp]
X_tmp = np.array([x.flatten() for x in X_tmp_resized])
y_tmp = np.array(y_tmp)
X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.2)

for parametre in parameter:
    print(f"==>Paramètre : {parametre}")
    # Récupération des paramètre de la liste "parametre"
    penalty = parametre[0]
    solver = parametre[1]
    l1_ratio = parametre[2]

    # Instanciation du modèle
    clf = linear_model.LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver=solver,
    )

    if args.train:
        print("==>==>Entrainement Model")
        # Entrainement du modèle de régréssion logistique
        clf.fit(X_train, y_train)

        # Enregistrement du modèle de régréssion logistique
        modele_name_file = "models/" + model_type + "_" + penalty + "_" + solver
        modele_name_file = modele_name_file
        with open(modele_name_file, 'wb') as fichier:
            pickle.dump(clf, fichier)

    if args.eval:
        print("==>==>Evaluation Model")
        with open(f'models/regression_logistique_{penalty}_{solver}',
                  'rb') as fichier:
            clf = pickle.load(fichier)

        # Prédiction du modèle de régréssion logistique
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        # Génération de la matrice de confusion et des métriques
        cm = confusion_matrix(y_test, y_pred)
        table = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'],
                            colnames=['Classe prédite'])
        plt.figure(figsize=(6, 4))
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion")
        dir = "reports/figures/matrice_confusion_regression_logistique"
        plt.savefig(f"{dir}_{penalty}_{solver}.png")
        plt.show()

        raw_report: Any = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
        report: Dict[str, Any] = cast(Dict[str, Any], raw_report)
        
        if y_proba.shape[1] == 2:
            # Cas binaire
            y_proba_pos = y_proba[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba_pos)
        else:
            # Cas multi-classes
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class='ovr')
        results: Dict[str, float | str] = {
            "C": C,
            "max_iter": max_iter,
            "penalty": penalty,
            "solver": solver,
            "roc_auc": float(roc_auc),
        }
        results.update(
            {f"cm_{i}_{j}": cm[i, j] for i in range(
                cm.shape[0]) for j in range(cm.shape[1])})

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                metrics_dict: Dict[str, float] = metrics
                results[f"{label}_precision"] = float(
                    metrics_dict.get("precision", 0.0))
                results[f"{label}_recall"] = float(
                    metrics_dict.get("recall", 0.0))
                results[f"{label}_f1"] = float(
                    metrics_dict.get("f1-score", 0.0))

        # Ajout des métriques dans le dataframe df_results
        df_results = pd.concat([df_results,
                                pd.DataFrame([results])], ignore_index=True)


# Génération d'un histogramme du weighted avg_f1
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_results,
    x="nom_filtre",
    y="weighted avg_f1",
    palette="tab10"
)
plt.title("Meilleur weighted avg_f1 par Filtre et Catégorie", pad=15)
plt.xlabel("Filtre", labelpad=10)
plt.ylabel("Meilleur weighted avg_f1", labelpad=10)
plt.xticks(rotation=45)
plt.legend(title="Catégorie")
plt.tight_layout()
plt.savefig("reports/figures/result_regression_logistique.png")
plt.show()

# Sauvegarde dans CSV
data_dir = os.path.join("data", "Result")
output_file = os.path.join(data_dir, "results_regression_logistique.csv")

if os.path.exists("reports/csv_export/results_regression_logistique.csv"):
    df_results.to_csv("reports/csv_export/results_regression_logistique.csv",
                      mode="a", index=False, sep=";")
else:
    df_results.to_csv("reports/csv_export/results_regression_logistique.csv",
                      index=False, sep=";")
