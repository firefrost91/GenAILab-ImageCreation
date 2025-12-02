# Generative AI Lab - Product Analysis and Image Generation

This project implements a complete AI agentic workflow for analyzing product descriptions and customer reviews using LLMs, and generating product images using diffusion models.

## Project Structure

```
.
├── workflow_orchestrator.py    # AI Agentic workflow orchestrator (main entry point)
├── main.py                      # Simple wrapper for workflow orchestrator
├── q1_data_collection.py       # Product selection and data collection
├── q2_llm_analysis.py          # LLM-based analysis with prompt engineering and RAG
├── q3_image_generation.py      # Image generation with DALL-E and Stable Diffusion
├── analysis_comparison.py      # Optional: Comparison and analysis of generated images
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── product_metadata.json       # Product metadata with review CSV references
├── *_reviews.csv               # Review CSV files (stanley, airpods, lego)
├── data/                       # Collected product data (generated)
├── results/                    # Analysis results and reports (generated)
└── generated_images/           # Generated product images (generated)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
STABILITY_API_KEY=your_stability_api_key_here  # Optional
```

### 3. Prepare Your Data

1. **Product Metadata** (`product_metadata.json`):
   ```json
   {
     "products": [
       {
         "name": "Product Name",
         "review_csv": "reviews.csv",
         "url": "https://...",
         "description": "Product description..."
       }
     ]
   }
   ```

2. **Review CSV Files**: Place CSV files with columns:
   - `Product`: Product name (should match or be similar to JSON name)
   - `Review_Text`: Full review text
   - Other columns (Rating, Author, Date, etc.) are optional

### 4. Run the Workflow

**Complete workflow:**
```bash
python main.py
```

**Or use the orchestrator directly:**
```bash
python workflow_orchestrator.py
```

**Skip specific steps:**
```bash
python workflow_orchestrator.py --skip q3_image_generation
```

## Workflow Overview

The workflow consists of three main steps:

### Q1: Data Collection
- Reads product metadata from `product_metadata.json`
- Loads reviews from CSV files specified in metadata
- Uses fuzzy matching to match product names between JSON and CSV
- Outputs standardized JSON files in `data/`

### Q2: LLM Analysis
- Analyzes product descriptions and reviews using GPT-4o-mini
- Uses both prompt engineering and RAG (Retrieval-Augmented Generation)
- Extracts visual characteristics, sentiment, topics, and image generation guidance
- Outputs analysis JSON files in `results/`

### Q3: Image Generation
- Generates product images using DALL-E 3 and Stable Diffusion
- Creates multiple variations per product
- Saves images and metadata in `generated_images/`

## Workflow Features

- **Automatic Dependency Management**: Validates prerequisites before each step
- **Error Handling**: Comprehensive error tracking and recovery
- **Progress Tracking**: Real-time logging with timestamps
- **Flexible Execution**: Can skip steps or run individual components
- **Report Generation**: Creates workflow reports in `results/workflow_report.json`

## Output Structure

After running the workflow:

```
data/
├── product_1_data.json
├── product_2_data.json
└── product_3_data.json

results/
├── product_1_analysis.json
├── product_2_analysis.json
├── product_3_analysis.json
└── workflow_report.json

generated_images/
├── product_1/
│   ├── product_1_dalle_1.png
│   ├── product_1_dalle_2.png
│   ├── product_1_dalle_3.png
│   └── product_1_generation_results.json
└── ...
```

## Running Individual Steps

You can also run individual steps if needed:

### Q1 Only
```python
from q1_data_collection import ProductDataCollector
collector = ProductDataCollector()
collector.collect_all_products("product_metadata.json")
```

### Q2 Only
```python
from q2_llm_analysis import analyze_all_products
from config import OPENAI_API_KEY
analyze_all_products(api_key=OPENAI_API_KEY)
```

### Q3 Only
```python
from q3_image_generation import generate_all_product_images
from config import OPENAI_API_KEY, STABILITY_API_KEY
generate_all_product_images(
    openai_api_key=OPENAI_API_KEY,
    stability_api_key=STABILITY_API_KEY
)
```

## Configuration

Edit `config.py` to customize:
- LLM model selection
- Chunking parameters
- Image generation settings
- Directory paths

## Troubleshooting

### Common Issues

1. **"No reviews found for product"**
   - Check that CSV file exists and is in the same directory as `product_metadata.json`
   - Verify product names match (fuzzy matching handles minor differences)
   - Check CSV has `Product` and `Review_Text` columns

2. **"OpenAI API key not found"**
   - Set `OPENAI_API_KEY` in `.env` file
   - Or pass it directly to the orchestrator

3. **"No analysis files found"**
   - Ensure Q1 completed successfully
   - Check that `data/*_data.json` files exist

## Documentation

- **WORKFLOW_GUIDE.md**: Detailed guide on using the workflow orchestrator
- **README.md**: This file - project overview and quick start

## Requirements

- Python 3.8+
- OpenAI API key (required)
- Stability AI API key (optional, for Stable Diffusion)
- See `requirements.txt` for Python packages

## License

This project is for educational purposes as part of the Generative AI Lab course.
