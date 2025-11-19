import os
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "unet_efficientnetv2_s_6band_finetuned_optimized_new_new.h5"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / MODEL_NAME


def ensure_models_directory():
    """Ensure models directory exists"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Models directory ready: {MODEL_DIR}")


def download_model_from_kaggle():
    """Download ML model from Kaggle using kagglehub if not present locally"""
    ensure_models_directory()

    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✅ Model already exists at {MODEL_PATH} ({file_size:.2f} MB)")
        return str(MODEL_PATH)

    # Set Kaggle credentials from environment variables BEFORE importing kagglehub
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        logger.error("❌ KAGGLE_USERNAME or KAGGLE_KEY not set in .env file")
        logger.error("💡 Please copy .env.example to .env and add your Kaggle API credentials")
        logger.error("💡 Get credentials from: https://www.kaggle.com/settings/account -> Create New API Token")
        logger.warning("⚠️ Continuing without model - inference will fail if needed")
        return str(MODEL_PATH)
    
    # Set credentials as environment variables (required by kagglehub)
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    logger.info(f"✅ Kaggle credentials loaded for user: {kaggle_username}")

    try:
        import kagglehub
        import shutil
        from pathlib import Path

        kaggle_model_path = os.getenv("KAGGLE_MODEL_PATH", "soumyadiptadey/khanannetra-production/tensorFlow2/version1")

        logger.info(f"📥 Downloading model from Kaggle: {kaggle_model_path}")

        # Download using kagglehub (automatically caches in MODEL_CACHE_DIR)
        downloaded_path = kagglehub.model_download(kaggle_model_path)
        logger.info(f"✅ Model downloaded to: {downloaded_path}")

        # Find and copy the .h5 file to our models directory
        downloaded_dir = Path(downloaded_path)
        h5_files = list(downloaded_dir.rglob("*.h5"))

        if h5_files:
            # Copy first .h5 file found to our standard location
            src_h5 = h5_files[0]
            logger.info(f"📋 Found model file: {src_h5}")
            shutil.copy2(src_h5, MODEL_PATH)
            logger.info(f"✅ Model copied to {MODEL_PATH}")
            return str(MODEL_PATH)
        else:
            logger.error(f"❌ No .h5 files found in {downloaded_path}")
            return str(MODEL_PATH)

    except ImportError:
        logger.warning("⚠️ kagglehub not installed. Falling back to kaggle-api...")
        return download_model_with_kaggle_api()
    except Exception as e:
        logger.error(f"❌ Failed to download model: {str(e)}")
        logger.warning(f"⚠️ Continuing without model - inference will fail if needed")
        return str(MODEL_PATH)


def download_model_with_kaggle_api():
    """Fallback: Download using old kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        kaggle_username = os.getenv("KAGGLE_USERNAME")
        kaggle_key = os.getenv("KAGGLE_KEY")

        if not kaggle_username or not kaggle_key:
            logger.warning("⚠️ KAGGLE_USERNAME or KAGGLE_KEY not set. Skipping model download.")
            return str(MODEL_PATH)

        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

        kaggle_model_id = os.getenv("KAGGLE_MODEL_ID", "soumyadiptadey/khanannetra-production")

        logger.info(f"📥 Downloading model using kaggle-api: {kaggle_model_id}")

        api = KaggleApi()
        api.authenticate()
        api.model_download(kaggle_model_id, path=str(MODEL_DIR))

        logger.info(f"✅ Model downloaded successfully to {MODEL_DIR}")
        return str(MODEL_PATH)

    except Exception as e:
        logger.error(f"❌ Kaggle API download failed: {str(e)}")
        return str(MODEL_PATH)


def get_model_path():
    """Get model path, downloading from Kaggle if necessary"""
    return download_model_from_kaggle()


# Auto-download on import if ENV variable is set
if os.getenv("DOWNLOAD_MODELS_ON_STARTUP", "true").lower() == "true":
    try:
        logger.info("🚀 Auto-downloading model on startup...")
        get_model_path()
    except Exception as e:
        logger.error(f"Model download failed on startup: {e}")
