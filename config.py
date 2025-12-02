"""
Configuration file for the project
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"  # Using GPT-4o-mini for cost efficiency
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Image Generation Configuration
DALLE_MODEL = "dall-e-3"
STABLE_DIFFUSION_MODEL = "stable-diffusion-xl-1024-v1-0"

# Output directories
DATA_DIR = "data"
IMAGES_DIR = "generated_images"
RESULTS_DIR = "results"

