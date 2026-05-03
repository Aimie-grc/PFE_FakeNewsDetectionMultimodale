"""Interface Streamlit → API FastAPI multipart /predict (lancer d’abord : uvicorn serve_multipart:app --port 8001).
puis 'streamlit run streamlit_app_multipart.py' pour lancer l'interface streamlit."""
import csv
import math
import os
import random
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="Détecteur de fake news",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    [data-testid='stSidebar'], [data-testid='stSidebarNav'], [data-testid='collapsedControl'] {display:none!important;}
    .result-ok {
        border-left: 6px solid #16a34a;
        padding: .8rem 1rem;
        border-radius: 8px;
        background: rgba(22, 163, 74, .08);
    }
    .result-warn {
        border-left: 6px solid #dc2626;
        padding: .8rem 1rem;
        border-radius: 8px;
        background: rgba(220, 38, 38, .08);
    }
    .muted {
        opacity: .8;
        font-size: .95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Détecteur de fake news multimodal")
st.caption("Analyse un texte et une image, puis affiche la classe prédite et son niveau de confiance.")

_API = os.environ.get("PREDICT_API_URL", "http://127.0.0.1:8001/predict")
# Affichage : 1 = vrai / fiable (vert), 0 = faux / fake (rouge) — renumérotation depuis la convention API (0=fiable, 1=fake).
_LABELS = {1: "Information fiable", 0: "Potentielle fake news"}
_ROOT = Path(__file__).parent
_TEST_CSV = _ROOT / "test.csv"


def _renumber_for_ui(data: dict) -> dict:
    """API renvoie 0=fiable, 1=fake ; l’UI affiche 1=fiable (vert), 0=fake (rouge)."""
    out = dict(data)
    logits = out.get("logits")
    if isinstance(logits, list) and len(logits) == 2:
        out["logits"] = [logits[1], logits[0]]
    c = out.get("class")
    if c is None:
        return out
    try:
        ci = int(c)
        if ci in (0, 1):
            out["class"] = 1 - ci
    except (TypeError, ValueError):
        pass
    return out


def _softmax(logits):
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(v - max_logit) for v in logits]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def _render_prediction(data):
    data = _renumber_for_ui(data)
    pred_class = int(data.get("class", -1))
    logits = data.get("logits", [])
    probs = _softmax(logits) if isinstance(logits, list) else []
    confidence = probs[pred_class] if probs and 0 <= pred_class < len(probs) else None
    label = _LABELS.get(pred_class, f"Classe {pred_class}")
    is_fiable = pred_class == 1  # 1 = vrai / fiable → vert

    css_class = "result-ok" if is_fiable else "result-warn"
    emoji = "✅" if is_fiable else "⚠️"
    st.markdown(
        f"""
        <div class="{css_class}">
            <h4 style="margin:0;">{emoji} Résultat: {label}</h4>
            <p class="muted" style="margin:.35rem 0 0 0;">
                Classe numérique: <b>{pred_class}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if confidence is not None:
            st.metric("Confiance du modèle", f"{confidence * 100:.1f}%")
            st.progress(max(0.0, min(float(confidence), 1.0)))
        else:
            st.metric("Confiance du modèle", "N/A")
    with c2:
        if probs:
            for idx, p in enumerate(probs):
                st.write(f"Classe {idx} - {_LABELS.get(idx, idx)}: **{p * 100:.1f}%**")

    fc = data.get("face_coherence")
    me = data.get("model_explain") or {}
    with st.expander("Explicabilité"):
        st.markdown("##### Classifieur (texte + image)")
        if me.get("error"):
            st.caption(f"Explication heuristique du modèle indisponible : {me['error']}")
        elif me.get("reason_auto"):
            st.write(me["reason_auto"])
            cont = me.get("contributions")
            if cont:
                parts = ", ".join(f"**{k}** {v} %" for k, v in cont.items())
                st.caption(f"Contributions estimées (heuristique) : {parts}")
        else:
            st.caption("Pas de `model_explain` dans la réponse API (redémarre le serveur si besoin).")

        st.markdown("##### Cohérence des personnes (image / texte)")
        if fc is None:
            st.caption(
                "Analyse de cohérence personnes (image / texte) non incluse dans la réponse API. "
                "Active côté serveur avec `ENABLE_FACE_COHERENCE=1` (par défaut) et redémarre l’API."
            )
        elif fc.get("error"):
            st.warning(f"Impossible de calculer la cohérence personnes : {fc['error']}")
        else:
            pct = fc.get("score_percent")
            n_text = (fc.get("details") or {}).get("nb_personnes_texte")
            if n_text == 0:
                st.info(
                    "Aucun nom de personne clairement extrait du texte : le score de cohérence n’est pas interprétable "
                    "(0 % par défaut)."
                )
            if pct is not None:
                st.markdown(
                    f"Il y a **{pct} %** de cohérence entre les personnes visibles sur l’image et les personnes "
                    "citées dans le texte (analyse Qwen2‑VL + recoupement des noms)."
                )
            pe = fc.get("parse_error")
            if pe:
                st.caption(f"Note analyse visuelle (JSON) : {pe}")
            personnes = fc.get("personnes")
            if personnes:
                st.markdown("**Personnes (image / texte / les deux) :**")
                for nom, statut in personnes.items():
                    st.write(f"- **{nom}** — {statut}")

    with st.expander("Détail du traitement"):
        lang = data.get("language")
        if lang:
            st.markdown(f"**Langue détectée :** `{lang}`")
        else:
            st.markdown(
                "**Langue détectée :** non déterminée — le texte est passé tel quel au modèle (anglais attendu)."
            )
        if data.get("text_translated") is True:
            st.markdown(
                "**Traduction :** oui — le texte a été traduit en **anglais** côté API avant la prédiction."
            )
        else:
            st.markdown(
                "**Traduction :** non — texte déjà en anglais, ou traduction non appliquée (voir note ci‑dessous si besoin)."
            )
        note = data.get("translation_note")
        if note:
            st.info(str(note))

    with st.expander("Détails techniques (API)"):
        st.json(data)


def _load_test_rows():
    if not _TEST_CSV.exists():
        return []
    with _TEST_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (r.get("content") or "").strip() and (r.get("image") or "").strip()]


def _resolve_image_path(image_value):
    p = Path(str(image_value).strip())
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend(
            [
                _ROOT / p,
                _ROOT / "images" / p.name,
                _ROOT / "data" / p.name,
                _ROOT / "data" / "images" / p.name,
            ]
        )
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _pick_random_example():
    rows = _load_test_rows()
    if not rows:
        return None, "Aucun exemple trouvé dans test.csv."
    sample = random.choice(rows)
    image_path = _resolve_image_path(sample.get("image", ""))
    if image_path is None:
        return None, (
            "Exemple trouvé dans test.csv mais image introuvable. "
            "Place les images dans le même dossier que streamlit_app_multipart.py ou ajuste le chemin."
        )
    return {
        "text": (sample.get("content") or "").strip(),
        "image_bytes": image_path.read_bytes(),
        "image_name": image_path.name,
        "image_path": str(image_path),
        "label": (sample.get("label") or "").strip(),
    }, None


if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""
if "sample_image_bytes" not in st.session_state:
    st.session_state.sample_image_bytes = None
if "sample_image_name" not in st.session_state:
    st.session_state.sample_image_name = ""
if "sample_label" not in st.session_state:
    st.session_state.sample_label = ""
if "sample_source" not in st.session_state:
    st.session_state.sample_source = ""

left, right = st.columns([1.15, 1])
with left:
    st.caption("Mode manuel ou test set Gossip (`test.csv`).")
    b1, b2 = st.columns(2)
    with b1:
        charger_random = st.button("Exemple aléatoire (test.csv)", width="stretch")
    with b2:
        vider_random = st.button("Vider l'exemple chargé", width="stretch")

    if charger_random:
        sample, err = _pick_random_example()
        if err:
            st.warning(err)
        else:
            st.session_state.sample_text = sample["text"]
            st.session_state.sample_image_bytes = sample["image_bytes"]
            st.session_state.sample_image_name = sample["image_name"]
            st.session_state.sample_label = sample["label"]
            st.session_state.sample_source = sample["image_path"]
            st.success("Exemple aléatoire chargé.")

    if vider_random:
        st.session_state.sample_text = ""
        st.session_state.sample_image_bytes = None
        st.session_state.sample_image_name = ""
        st.session_state.sample_label = ""
        st.session_state.sample_source = ""

    default_text = st.session_state.sample_text if st.session_state.sample_text else ""
    texte = st.text_area(
        "Texte",
        placeholder="Saisis ici le texte associé à l'image...",
        height=180,
        help="Le texte est tronqué à 512 tokens côté API.",
        value=default_text,
    )
    fichier = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if st.session_state.sample_image_bytes is not None:
        st.caption(
            f"Exemple chargé: `{st.session_state.sample_image_name}` "
            f"(label attendu: {st.session_state.sample_label or 'N/A'})"
        )
    image_for_predict = fichier.getvalue() if fichier else st.session_state.sample_image_bytes
    image_name = fichier.name if fichier else st.session_state.sample_image_name
    lancer = st.button(
        "Lancer la prédiction",
        type="primary",
        width="stretch",
        disabled=not (image_for_predict and texte and texte.strip()),
    )

with right:
    st.subheader("Aperçu")
    if image_for_predict:
        st.image(image_for_predict, caption=f"Fichier: {image_name}", width="stretch")
        if st.session_state.sample_source and not fichier:
            st.caption(f"Source: `{st.session_state.sample_source}`")
    else:
        st.info("Ajoute une image pour afficher l'aperçu.")

if lancer:
    if not image_for_predict:
        st.warning("Choisis une image.")
    elif not texte or not texte.strip():
        st.warning("Saisis un texte.")
    else:
        try:
            with st.spinner("Prédiction en cours..."):
                response = requests.post(
                    _API,
                    data={"text": texte.strip()},
                    files={"image": (image_name or "image.jpg", image_for_predict, "application/octet-stream")},
                    timeout=300,
                )
            if response.status_code != 200:
                st.error(f"Erreur HTTP {response.status_code}: {response.text}")
            else:
                _render_prediction(response.json())
        except requests.RequestException as exc:
            st.error(
                f"Impossible de joindre l'API ({exc}). "
                "Vérifie que `uvicorn serve_multipart:app --port 8001` (ou équivalent) tourne et que l'URL est correcte."
            )
