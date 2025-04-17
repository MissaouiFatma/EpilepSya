import pandas as pd
import numpy as np
import streamlit as st
import base64
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import kurtosis, skew
import os
import bcrypt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# User database
USERS_DB = "users.csv"
RECORDS_DB = "records.csv" # Nouveau fichier pour les enregistrements
ADMIN_PASSWORD = "adminsession" # Remplacez par un mot de passe fort

def load_image(image_path):
    """Load an image from a path and convert it to base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_path):
    """Set a background image using HTML and CSS."""
    image_base64 = load_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{image_base64}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            height: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_users():
    """Load registered users."""
    try:
        return pd.read_csv(USERS_DB, encoding='utf-8')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Nom', 'Prénom', 'CIN', 'ID', 'Password'])

def save_user(nom, prenom, cin, user_id, password):
    """Save a new user if all fields are filled (password in plain text)."""
    users = load_users()
    new_user = pd.DataFrame([[nom, prenom, cin, user_id, password]], columns=['Nom', 'Prénom', 'CIN', 'ID', 'Password'])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USERS_DB, index=False, encoding='utf-8')

def authenticate(user_id, password):
    """Check user authentication (password in plain text)."""
    users = load_users()
    user_row = users[users['ID'] == user_id]
    if not user_row.empty:
        stored_password = user_row.iloc[0]['Password']
        return password == stored_password
    return False

def load_records():
    """Load the records database."""
    try:
        return pd.read_csv(RECORDS_DB, encoding='utf-8')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Nom', 'Prénom', 'ID', 'Password (Plain)', 'CSV Uploaded', 'Prediction Result'])

def save_record(user_id, csv_filename, prediction_result):
    """Save a new record."""
    users = load_users()
    user_data = users[users['ID'] == user_id].iloc[0]
    records = load_records()
    new_record = pd.DataFrame([[user_data['Nom'], user_data['Prénom'], user_data['ID'], user_data['Password'], csv_filename, prediction_result]],
                              columns=['Nom', 'Prénom', 'ID', 'Password (Plain)', 'CSV Uploaded', 'Prediction Result'])
    records = pd.concat([records, new_record], ignore_index=True)
    records.to_csv(RECORDS_DB, index=False, encoding='utf-8')

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = None

def load_data(csv_file):
    """Charge le fichier CSV contenant les signaux EEG bruts."""
    return pd.read_csv(csv_file)

def compute_derivatives(features, n):
    """Calcule les dérivées d'ordre n."""
    return np.diff(features, n=n, axis=1)

def extract_features(d1, d2, d3):
    """Extrait les 7 caractéristiques principales de chaque ordre de dérivée."""
    all_features = []

    def get_stats(data):
        min_vals = np.min(data, axis=1).reshape(-1, 1)
        max_vals = np.max(data, axis=1).reshape(-1, 1)
        mean_vals = np.mean(data, axis=1).reshape(-1, 1)
        median_vals = np.median(data, axis=1).reshape(-1, 1)
        std_vals = np.std(data, axis=1).reshape(-1, 1)
        kurtosis_vals = kurtosis(data, axis=1).reshape(-1, 1)
        skewness_vals = skew(data, axis=1).reshape(-1, 1)
        return np.hstack([min_vals, max_vals, mean_vals, std_vals, kurtosis_vals, median_vals, skewness_vals])

    if d1.size > 0:
        all_features.append(get_stats(d1))
    if d2.size > 0:
        all_features.append(get_stats(d2))
    if d3.size > 0:
        all_features.append(get_stats(d3))

    return np.hstack(all_features) if all_features else np.array([])
def preprocess_data(df):
    """Prétraite les données : calcul des dérivées et extraction de caractéristiques."""
    if df.shape[1] < 3:
        raise ValueError(f"Le DataFrame doit avoir au moins 3 colonnes. Il a {df.shape[1]} colonnes.")
    features = df.iloc[:, 2:].values
    if features.size == 0:
        raise ValueError("Le tableau 'features' (données brutes) est vide.")
    d1 = compute_derivatives(features, n=1)
    d2 = compute_derivatives(features, n=2)
    d3 = compute_derivatives(features, n=3)
    engineered_features = extract_features(d1, d2, d3)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engineered_features)
    return scaled_features, df.iloc[:, 0].values


def train_and_select_best_model(X, y):
    """Entraîne plusieurs modèles et sélectionne le plus performant."""
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'DecisionTree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier()
    }
    model_scores = {}
    best_model, best_score, best_model_name = None, 0, ''
    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)
        score = accuracy_score(y, predictions)
        model_scores[name] = score
        if score > best_score:
            best_score, best_model, best_model_name = score, model, name
    return best_model, best_model_name, model_scores

def predict(model, data):
    """Effectue la prédiction avec le modèle donné."""
    return model.predict(data)

def show_home_page():
    """Page d'accueil : Connexion ou Création de compte."""
    st.markdown('<h1 style="font-size: 155px;">EpilepSya</h1>', unsafe_allow_html=True)

    if st.button("Se connecter"):
        st.session_state["page"] = "login"
        st.experimental_rerun()

    if st.button("Créer un compte"):
        st.session_state["page"] = "register"
        st.experimental_rerun()

def show_login_page():
    """Page de connexion."""
    st.subheader("Connexion")
    user_id = st.text_input("ID")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if authenticate(user_id, password):
            st.session_state["authenticated"] = True
            st.session_state["user_id"] = user_id
            st.session_state["page"] = "upload"
            st.experimental_rerun()
        else:
            st.error("Identifiants incorrects.")

    if st.button("Retour à l'accueil"):
        st.session_state["page"] = "home"
        st.experimental_rerun()

def show_register_page():
    """Page de création de compte."""
    st.subheader("Créer un nouveau compte")
    nom = st.text_input("Nom")
    prenom = st.text_input("Prénom")
    cin = st.text_input("CIN")
    user_id = st.text_input("ID")
    password = st.text_input("Mot de passe", type="password")

    if st.button("S'inscrire"):
        save_user(nom, prenom, cin, user_id, password)
        st.success("Compte créé avec succès!")
        st.session_state["page"] = "login"
        st.experimental_rerun()

    if st.button("Retour à l'accueil"):
        st.session_state["page"] = "home"
        st.experimental_rerun()

def show_upload_predict_page():
    """Page de téléversement et prédiction des données EEG (uniquement pour les utilisateurs connectés)."""
    st.title("Téléverser un fichier CSV pour analyse EEG")

    if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
        st.error("Vous devez être connecté pour analyser les données.")
        return

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"], on_change=lambda: st.session_state.update(uploaded_file=uploaded_file, predictions=None, best_model_name=None, model_scores=None))

    if uploaded_file is not None:
        st.session_state["uploaded_file"] = uploaded_file
        st.write("Fichier chargé avec succès!")
        if st.button("Analyser"):
            try:
                df = load_data(st.session_state["uploaded_file"])
                st.write(f"Nombre de lignes dans le DataFrame chargé : {len(df)}")
                st.write(f"Nombre de colonnes dans le DataFrame chargé : {df.shape[1]}")
                st.write("Prétraitement des données...")
                data, labels = preprocess_data(df)
                st.write(f"Taille du tableau 'data' après prétraitement : {data.shape}")
                st.write("Sélection des modèles...")
                best_model, best_model_name, model_scores = train_and_select_best_model(data, labels)
                st.session_state["best_model_name"] = best_model_name
                st.session_state["model_scores"] = model_scores

                # Visualisation des scores
                fig, ax = plt.subplots()
                model_names = list(model_scores.keys())
                accuracies = list(model_scores.values())
                colors = plt.cm.viridis(np.linspace(0, 1, len(model_names))) # Générer des couleurs

                ax.pie(accuracies, labels=None, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

                # Légende colorée avec les valeurs d'accuracy
                legend_patches = [mpatches.Patch(color=colors[i], label=f'{model_names[i]}: {accuracies[i]:.4f}') for i in range(len(model_names))]
                plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                st.pyplot(plt) # Afficher la légende séparément (peut nécessiter un deuxième appel à st.pyplot)

                st.write(f"Meilleur modèle sélectionné : {st.session_state['best_model_name']}")
                st.write("Exécution de la prédiction...")
                predictions = predict(best_model, data)
                st.session_state["predictions"] = predictions
                st.write("Résultats des prédictions :")
                st.write(pd.DataFrame(st.session_state["predictions"], columns=['Prédiction']))
                st.success("Analyse terminée!")
                save_record(st.session_state["user_id"], st.session_state["uploaded_file"].name, " ".join(map(str, st.session_state["predictions"]))) # Enregistrer l'enregistrement
            except ValueError as ve:
                st.error(f"Erreur lors de l'analyse : {ve}")
            except Exception as e:
                st.error(f"Une erreur inattendue est survenue lors de l'analyse : {e}")

def show_records_page():
    """Page pour afficher les enregistrements, protégée par mot de passe."""
    st.title("Enregistrements des Utilisateurs (Admin Only)")
    admin_password_input = st.text_input("Mot de passe administrateur :", type="password")

    if admin_password_input == ADMIN_PASSWORD:
        records = load_records()
        st.dataframe(records)
        if st.button("Retour à l'accueil"):
            st.session_state["page"] = "home"
            st.experimental_rerun()
    elif admin_password_input:
        st.error("Mot de passe administrateur incorrect.")
    else:
        st.info("Veuillez entrer le mot de passe administrateur pour afficher les enregistrements.")

def streamlit_interface():
    """Main interface for Streamlit application."""
    set_background("C:\\Users\\asus\\Downloads\\brain.webp")

    if st.session_state["page"] == "home":
        show_home_page()
    elif st.session_state["page"] == "login":
        show_login_page()
    elif st.session_state["page"] == "register":
        show_register_page()
    elif st.session_state["page"] == "upload" and st.session_state.get("authenticated", False):
        show_upload_predict_page()
    elif st.session_state["page"] == "records" and st.session_state.get("authenticated", False):
        show_records_page()

    if st.session_state.get("authenticated", False) and st.session_state["page"] != "records":
        if st.sidebar.button("Enregistrements"):
            st.session_state["page"] = "records"
            st.experimental_rerun()


if __name__ == "__main__":
    streamlit_interface()