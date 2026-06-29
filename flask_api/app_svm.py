from flask import Flask, request, jsonify, render_template
from pathlib import Path
import joblib
import pickle
import logging
import re

# =====================================================
# PROJECT PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_DIR = BASE_DIR
RECOMMENDER_DIR = PROJECT_ROOT / "recommender"
TEMPLATE_DIR = PROJECT_ROOT / "templates"

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =====================================================
# FLASK APP
# =====================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR)
)

# =====================================================
# LOAD SVM MODEL
# =====================================================

logger.info("=" * 60)
logger.info("Loading Sentiment Model...")
logger.info("=" * 60)

try:

    svm_model = joblib.load(
        MODEL_DIR / "svm_model.pkl"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    logger.info("Sentiment model loaded successfully.")

except Exception as e:

    logger.exception("Failed to load sentiment model.")
    raise e

# =====================================================
# LOAD RECOMMENDER MATRICES
# =====================================================

logger.info("=" * 60)
logger.info("Loading Recommendation Matrices...")
logger.info("=" * 60)

try:

    with open(
        RECOMMENDER_DIR / "content_similarity_matrix.pkl",
        "rb"
    ) as f:

        content_matrix = pickle.load(f)

    with open(
        RECOMMENDER_DIR / "item_similarity_matrix.pkl",
        "rb"
    ) as f:

        collaborative_matrix = pickle.load(f)

    logger.info("Recommendation matrices loaded successfully.")

except Exception as e:

    logger.exception("Failed to load recommendation matrices.")
    raise e

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# =====================================================
# RECOMMENDATION FUNCTIONS
# =====================================================

def get_content_recommendations(
    product_id,
    top_n=5
):

    if product_id not in content_matrix.index:
        return []

    scores = (
        content_matrix.loc[product_id]
        .sort_values(
            ascending=False
        )
    )

    recommendations = (
        scores
        .drop(product_id)
        .head(top_n)
        .index
        .tolist()
    )

    return recommendations


def get_collaborative_recommendations(
    product_id,
    top_n=5
):

    if product_id not in collaborative_matrix.index:
        return []

    scores = (
        collaborative_matrix.loc[product_id]
        .sort_values(
            ascending=False
        )
    )

    recommendations = (
        scores
        .drop(product_id)
        .head(top_n)
        .index
        .tolist()
    )

    return recommendations


def build_hybrid_recommendations(
    product_id,
    top_n=5
):

    content_rec = get_content_recommendations(
        product_id,
        top_n
    )

    collaborative_rec = get_collaborative_recommendations(
        product_id,
        top_n
    )

    hybrid = []

    max_len = max(
        len(content_rec),
        len(collaborative_rec)
    )

    for i in range(max_len):

        if i < len(content_rec):

            if content_rec[i] not in hybrid:

                hybrid.append(
                    content_rec[i]
                )

        if i < len(collaborative_rec):

            if collaborative_rec[i] not in hybrid:

                hybrid.append(
                    collaborative_rec[i]
                )

    return hybrid[:top_n]
# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================================
# SENTIMENT API
# =====================================================

@app.route(
    "/api/sentiment",
    methods=["POST"]
)
def predict_sentiment():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({

                "status": "error",

                "message": "Invalid JSON request."

            }), 400

        text = data.get(
            "text",
            ""
        ).strip()

        if text == "":

            return jsonify({

                "status": "error",

                "message": "Text is required."

            }), 400

        cleaned_text = clean_text(text)

        vector = vectorizer.transform(

            [cleaned_text]

        )

        prediction = svm_model.predict(

            vector

        )

        sentiment = label_encoder.inverse_transform(

            prediction

        )[0]

        # =================================================
        # Recommendation Action
        # =================================================

        if sentiment.lower() == "positif":

            action = "special_for_you"

        else:

            action = "ignore"

        return jsonify({

            "status": "success",

            "text": text,

            "clean_text": cleaned_text,

            "sentiment": sentiment,

            "recommendation_action": action

        })

    except Exception as e:

        logger.exception(e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =====================================================
# CONTENT-BASED API
# =====================================================

@app.route(
    "/api/content-based/<int:product_id>"
)
def content_based_api(product_id):

    try:

        recommendations = get_content_recommendations(

            product_id

        )

        return jsonify({

            "status": "success",

            "method": "content_based",

            "product_id": product_id,

            "recommendations": recommendations

        })

    except Exception as e:

        logger.exception(e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =====================================================
# COLLABORATIVE API
# =====================================================

@app.route(
    "/api/collaborative/<int:product_id>"
)
def collaborative_api(product_id):

    try:

        recommendations = get_collaborative_recommendations(

            product_id

        )

        return jsonify({

            "status": "success",

            "method": "collaborative",

            "product_id": product_id,

            "recommendations": recommendations

        })

    except Exception as e:

        logger.exception(e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =====================================================
# HYBRID API
# =====================================================

@app.route(
    "/api/hybrid/<int:product_id>"
)
def hybrid_api(product_id):

    try:

        recommendations = build_hybrid_recommendations(

            product_id

        )

        return jsonify({

            "status": "success",

            "method": "hybrid",

            "product_id": product_id,

            "recommendations": recommendations

        })

    except Exception as e:

        logger.exception(e)

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500
# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/api/status")
def status():

    return jsonify({

        "application": "Gema Sandang AI Service",

        "model": "Support Vector Machine (SVM)",

        "status": "running",

        "version": "1.0"

    })


# =====================================================
# ROOT INFORMATION
# =====================================================

@app.route("/api")
def api_info():

    return jsonify({

        "application": "Gema Sandang AI Service",

        "version": "1.0",

        "available_endpoints": {

            "POST /api/sentiment":
                "Predict sentiment",

            "GET /api/content-based/<product_id>":
                "Content-Based Recommendation",

            "GET /api/collaborative/<product_id>":
                "Collaborative Recommendation",

            "GET /api/hybrid/<product_id>":
                "Hybrid Recommendation",

            "GET /api/status":
                "Health Check"

        }

    })


# =====================================================
# ERROR HANDLER
# =====================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "status": "error",

        "message": "Endpoint not found."

    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "status": "error",

        "message": "Internal server error."

    }), 500


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("GEMA SANDANG AI SERVICE")
    logger.info("=" * 60)

    logger.info("Model              : SVM")
    logger.info("Sentiment Analysis : Ready")
    logger.info("Recommendation     : Ready")
    logger.info("Content-Based      : Ready")
    logger.info("Collaborative      : Ready")
    logger.info("Hybrid             : Ready")

    logger.info("-" * 60)
    logger.info("Running Flask Server...")
    logger.info("URL : http://127.0.0.1:5000")
    logger.info("=" * 60)

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )