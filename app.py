import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Prédiction Résistance Béton",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Style CSS personnalisé ───────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
        cursor: pointer;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# CHARGEMENT ET ENTRAÎNEMENT DES MODÈLES
# ════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    df = pd.read_csv('Concrete_Data.csv', sep=';',
                     decimal=',', encoding='latin1')
    df.columns = [
        'ciment', 'laitier', 'cendres_volantes', 'eau',
        'superplastifiant', 'gros_granulats', 'granulats_fins',
        'age', 'resistance'
    ]
    return df

@st.cache_resource
def entrainer_modeles():
    df = charger_donnees()
    X = df.drop('resistance', axis=1)
    y = df['resistance']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    modeles = {
        'RLM': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf'),
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=20,
            min_samples_split=2, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=42, verbosity=0
        )
    }

    resultats = {}
    modeles_entraines = {}

    for nom, modele in modeles.items():
        if nom in ['RLM', 'KNN', 'SVR']:
            modele.fit(X_train_scaled, y_train)
            y_pred = modele.predict(X_test_scaled)
            modeles_entraines[nom] = (modele, scaler, True)
        else:
            modele.fit(X_train, y_train)
            y_pred = modele.predict(X_test)
            modeles_entraines[nom] = (modele, scaler, False)

        resultats[nom] = {
            'R²': round(r2_score(y_test, y_pred), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        }

    return modeles_entraines, resultats, X.columns.tolist()

# ════════════════════════════════════════════════════════════════════
# INTERFACE PRINCIPALE
# ════════════════════════════════════════════════════════════════════

# Titre
st.markdown('<div class="main-title">🏗️ Prédiction de la Résistance du Béton</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Application développée dans le cadre du PFE — '
            'Salma Rhomari | Construction LAB | FST Tanger</div>',
            unsafe_allow_html=True)

st.markdown("---")

# Chargement
with st.spinner("Chargement et entraînement des modèles..."):
    modeles_entraines, resultats, features = entrainer_modeles()

# ── Navigation ───────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "📌 Navigation",
    ["🔮 Prédiction", "📊 Tableau de bord", "📈 Comparaison des modèles"]
)

# ════════════════════════════════════════════════════════════════════
# PAGE 1 : PRÉDICTION
# ════════════════════════════════════════════════════════════════════

if page == "🔮 Prédiction":

    st.header("🔮 Prédire la Résistance d'un Béton")
    st.markdown("Entrez la composition de votre béton pour obtenir une "
                "estimation instantanée de sa résistance à la compression.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧱 Composition du mélange")
        ciment = st.slider("Ciment (kg/m³)", 100.0, 550.0, 300.0, 1.0)
        laitier = st.slider("Laitier de haut fourneau (kg/m³)", 0.0, 360.0, 50.0, 1.0)
        cendres = st.slider("Cendres volantes (kg/m³)", 0.0, 200.0, 0.0, 1.0)
        eau = st.slider("Eau (kg/m³)", 120.0, 250.0, 180.0, 1.0)

    with col2:
        st.subheader("⚗️ Adjuvants et granulats")
        superplast = st.slider("Superplastifiant (kg/m³)", 0.0, 35.0, 5.0, 0.1)
        gros_gran = st.slider("Gros granulats (kg/m³)", 800.0, 1150.0, 970.0, 1.0)
        gran_fins = st.slider("Granulats fins (kg/m³)", 590.0, 1000.0, 770.0, 1.0)
        age = st.slider("Âge du béton (jours)", 1, 365, 28, 1)

    st.markdown("---")

    # Sélection du modèle
    modele_choisi = st.selectbox(
        "🤖 Choisir le modèle de prédiction",
        list(modeles_entraines.keys()),
        index=4  # XGBoost par défaut
    )

    # Bouton de prédiction
    if st.button("🚀 Prédire la résistance"):
        entree = np.array([[ciment, laitier, cendres, eau,
                            superplast, gros_gran, gran_fins, age]])
        entree_df = pd.DataFrame(entree, columns=features)

        modele, scaler, normaliser = modeles_entraines[modele_choisi]

        if normaliser:
            entree_scaled = scaler.transform(entree_df)
            prediction = modele.predict(entree_scaled)[0]
        else:
            prediction = modele.predict(entree_df)[0]

        # Résultat
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            st.markdown(f"""
            <div class="result-box">
                🏗️ Résistance prédite<br>
                <span style="font-size:2.5rem">{prediction:.2f} MPa</span>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            r2 = resultats[modele_choisi]['R²']
            st.markdown(f"""
            <div class="success-box">
                📊 Fiabilité du modèle<br>
                <span style="font-size:2rem">R² = {r2}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_res3:
            rmse = resultats[modele_choisi]['RMSE']
            st.markdown(f"""
            <div class="metric-card">
                📏 Erreur moyenne<br>
                <span style="font-size:2rem">± {rmse:.2f} MPa</span>
            </div>
            """, unsafe_allow_html=True)

        # Interprétation
        st.markdown("---")
        st.subheader("📋 Interprétation")

        if prediction < 20:
            st.error(f"⚠️ Résistance faible ({prediction:.2f} MPa) — "
                     "Béton non structurel, à réviser.")
        elif prediction < 30:
            st.warning(f"🟡 Résistance correcte ({prediction:.2f} MPa) — "
                       "Béton courant B25.")
        elif prediction < 50:
            st.success(f"✅ Bonne résistance ({prediction:.2f} MPa) — "
                       "Béton structurel B30-B45.")
        else:
            st.success(f"🏆 Très haute résistance ({prediction:.2f} MPa) — "
                       "Béton haute performance.")

        # Prédictions de tous les modèles
        st.markdown("---")
        st.subheader("🔄 Prédictions de tous les modèles")

        preds_tous = {}
        for nom, (mod, sc, norm) in modeles_entraines.items():
            if norm:
                e = sc.transform(entree_df)
                preds_tous[nom] = round(mod.predict(e)[0], 2)
            else:
                preds_tous[nom] = round(mod.predict(entree_df)[0], 2)

        df_preds = pd.DataFrame({
            'Modèle': list(preds_tous.keys()),
            'Prédiction (MPa)': list(preds_tous.values()),
            'R²': [resultats[n]['R²'] for n in preds_tous.keys()],
            'RMSE (MPa)': [resultats[n]['RMSE'] for n in preds_tous.keys()]
        })
        st.dataframe(df_preds, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 : TABLEAU DE BORD
# ════════════════════════════════════════════════════════════════════

elif page == "📊 Tableau de bord":

    st.header("📊 Tableau de Bord — Performances des Modèles")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    couleurs = ['#2E86AB', '#A23B72', '#F18F01', '#44BBA4', '#C73E1D']

    for i, (nom, res) in enumerate(resultats.items()):
        with cols[i]:
            st.metric(
                label=f"R² — {nom}",
                value=res['R²'],
                delta=f"RMSE: {res['RMSE']} MPa"
            )

    st.markdown("---")

    # Tableau complet
    st.subheader("📋 Tableau Comparatif Complet")
    df_res = pd.DataFrame(resultats).T.reset_index()
    df_res.columns = ['Modèle', 'R²', 'MAE (MPa)', 'RMSE (MPa)']
    st.dataframe(df_res.style.highlight_max(subset=['R²'], color='#90EE90')
                             .highlight_min(subset=['MAE (MPa)', 'RMSE (MPa)'],
                                            color='#90EE90'),
                 use_container_width=True)

    st.markdown("---")

    # Graphiques
    st.subheader("📈 Visualisation des Performances")

    noms = list(resultats.keys())
    r2_vals = [resultats[n]['R²'] for n in noms]
    mae_vals = [resultats[n]['MAE'] for n in noms]
    rmse_vals = [resultats[n]['RMSE'] for n in noms]

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(noms, r2_vals, color=couleurs, edgecolor='white')
        ax.set_title('R²', fontweight='bold')
        ax.set_ylim(0, 1)
        for i, v in enumerate(r2_vals):
            ax.text(i, v + 0.01, str(v), ha='center', fontsize=9)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(noms, mae_vals, color=couleurs, edgecolor='white')
        ax.set_title('MAE (MPa)', fontweight='bold')
        for i, v in enumerate(mae_vals):
            ax.text(i, v + 0.05, str(v), ha='center', fontsize=9)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(noms, rmse_vals, color=couleurs, edgecolor='white')
        ax.set_title('RMSE (MPa)', fontweight='bold')
        for i, v in enumerate(rmse_vals):
            ax.text(i, v + 0.05, str(v), ha='center', fontsize=9)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Meilleur modèle
    st.markdown("---")
    meilleur = max(resultats, key=lambda x: resultats[x]['R²'])
    st.success(f"🏆 **Meilleur modèle : {meilleur}** avec R² = "
               f"{resultats[meilleur]['R²']} et RMSE = "
               f"{resultats[meilleur]['RMSE']} MPa")

# ════════════════════════════════════════════════════════════════════
# PAGE 3 : COMPARAISON
# ════════════════════════════════════════════════════════════════════

elif page == "📈 Comparaison des modèles":

    st.header("📈 Comparaison Détaillée des Modèles")

    df = charger_donnees()
    X = df.drop('resistance', axis=1)
    y = df['resistance']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    st.subheader("🎯 Valeurs Réelles vs Prédites")

    couleurs = ['#2E86AB', '#A23B72', '#F18F01', '#44BBA4', '#C73E1D']
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i, (nom, (mod, sc, norm)) in enumerate(modeles_entraines.items()):
        if norm:
            y_pred = mod.predict(X_test_scaled)
        else:
            y_pred = mod.predict(X_test)

        axes[i].scatter(y_test, y_pred, alpha=0.4,
                        color=couleurs[i], s=15)
        axes[i].plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()],
                     'r--', linewidth=1.5)
        axes[i].set_title(nom, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Réel (MPa)', fontsize=8)
        axes[i].set_ylabel('Prédit (MPa)', fontsize=8)

    plt.suptitle('Valeurs réelles vs prédites',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    # Feature Importance
    st.markdown("---")
    st.subheader("🔍 Importance des Variables")

    rf_mod = modeles_entraines['Random Forest'][0]
    xgb_mod = modeles_entraines['XGBoost'][0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    idx_rf = np.argsort(rf_mod.feature_importances_)
    axes[0].barh([features[i] for i in idx_rf],
                 rf_mod.feature_importances_[idx_rf],
                 color='#44BBA4', edgecolor='white')
    axes[0].set_title('Random Forest', fontweight='bold')
    axes[0].set_xlabel('Importance')

    idx_xgb = np.argsort(xgb_mod.feature_importances_)
    axes[1].barh([features[i] for i in idx_xgb],
                 xgb_mod.feature_importances_[idx_xgb],
                 color='#C73E1D', edgecolor='white')
    axes[1].set_title('XGBoost', fontweight='bold')
    axes[1].set_xlabel('Importance')

    plt.tight_layout()
    st.pyplot(fig)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem'>"
    "🏗️ Application développée par <b>Salma Rhomari</b> — PFE Licence SSD "
    "— FST Tanger 2025-2026"
    "</div>",
    unsafe_allow_html=True
)