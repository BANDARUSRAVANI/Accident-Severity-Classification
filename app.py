import os
import sys
import shutil
import datetime
from flask import Flask, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from db import init_db, close_db
    from routes.dataset import dataset_bp
    from routes.training import training_bp
    from routes.prediction import prediction_bp
    from routes.history import history_bp
except (ImportError, ValueError):
    from backend.db import init_db, close_db
    from backend.routes.dataset import dataset_bp
    from backend.routes.training import training_bp
    from backend.routes.prediction import prediction_bp
    from backend.routes.history import history_bp

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_data')

UPLOADED_DATASET_PATH = os.path.join(DATA_DIR, 'uploaded_dataset.csv')
SAMPLE_DATASET_PATH = os.path.join(SAMPLE_DIR, 'sample_accidents.csv')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')


def create_app():
    """Application factory for Flask backend."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Ensure required directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Initialize SQLite database
    init_db()

    # If uploaded dataset doesn't exist yet, populate with sample data if available
    if not os.path.exists(UPLOADED_DATASET_PATH) and os.path.exists(SAMPLE_DATASET_PATH):
        try:
            shutil.copyfile(SAMPLE_DATASET_PATH, UPLOADED_DATASET_PATH)
        except Exception:
            pass

    # Register blueprints
    app.register_blueprint(dataset_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)

    # Register teardown
    app.teardown_appcontext(close_db)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        dataset_loaded = os.path.exists(UPLOADED_DATASET_PATH)
        model_trained = os.path.exists(BEST_MODEL_PATH)

        return jsonify({
            "status": "ok",
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_loaded": dataset_loaded,
            "model_trained": model_trained,
            "version": "1.0.0"
        }), 200

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "error": "Not Found",
            "message": "The requested API endpoint was not found."
        }), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": str(e)
        }), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
