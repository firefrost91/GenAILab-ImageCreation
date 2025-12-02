# AI Agentic Workflow Guide

This guide explains how to use the AI Agentic Workflow Orchestrator to run the complete pipeline.

## Overview

The workflow orchestrator connects three main components:

1. **Q1: Data Collection** (`q1_data_collection.py`)
   - Reads product metadata from `product_metadata.json`
   - Loads reviews from CSV files specified in the metadata
   - Outputs standardized JSON files in `data/`

2. **Q2: LLM Analysis** (`q2_llm_analysis.py`)
   - Analyzes product descriptions and reviews using LLM
   - Uses both prompt engineering and RAG approaches
   - Outputs analysis JSON files in `results/`

3. **Q3: Image Generation** (`q3_image_generation.py`)
   - Generates product images using DALL-E 3 and Stable Diffusion
   - Creates images based on LLM analysis
   - Saves images in `generated_images/`

## Quick Start

### Basic Usage

Run the complete workflow:

```bash
python main.py
```

Or use the orchestrator directly:

```bash
python workflow_orchestrator.py
```

### Advanced Usage

Skip specific steps:

```bash
python workflow_orchestrator.py --skip q3_image_generation
```

Use custom paths:

```bash
python workflow_orchestrator.py \
    --products-json product_metadata.json \
    --data-dir data \
    --results-dir results \
    --images-dir generated_images
```

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────┐
│           WorkflowOrchestrator                          │
│  - Prerequisite Validation                             │
│  - Step Orchestration                                   │
│  - Error Handling & Recovery                            │
│  - Progress Tracking                                    │
│  - Report Generation                                    │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Q1: Data   │  │  Q2: LLM     │  │  Q3: Image   │
│  Collection  │→ │  Analysis    │→ │  Generation  │
└──────────────┘  └──────────────┘  └──────────────┘
        │               │               │
        ▼               ▼               ▼
   data/*.json    results/*.json   images/*/
```

## Input Format

### Product Metadata JSON

The `product_metadata.json` file should have this structure:

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

Each product must have:
- `name`: Product name (used for matching with CSV reviews)
- `review_csv`: Filename of the CSV file containing reviews
- `description`: Product description text
- `url`: (Optional) Product URL

### Review CSV Format

CSV files should have these columns:
- `Product`: Product name (must match JSON product name)
- `Review_Text`: Full review text
- Other columns (Rating, Author, Date, etc.) are optional

## Workflow Features

### 1. Prerequisite Validation
- Checks if product metadata JSON exists
- Validates OpenAI API key
- Verifies CSV files referenced in metadata exist

### 2. Error Handling
- Each step validates its output before proceeding
- Errors are logged and tracked
- Workflow can continue or stop based on step failures

### 3. Progress Tracking
- Real-time logging of each step
- Timestamped events
- Success/failure indicators

### 4. Report Generation
- Comprehensive workflow report saved to `results/workflow_report.json`
- Includes:
  - Step completion status
  - Files generated
  - Errors encountered
  - Duration metrics

## Output Structure

After running the workflow:

```
.
├── data/
│   ├── product_1_data.json
│   ├── product_2_data.json
│   └── product_3_data.json
├── results/
│   ├── product_1_analysis.json
│   ├── product_2_analysis.json
│   ├── product_3_analysis.json
│   └── workflow_report.json
└── generated_images/
    ├── product_1/
    │   ├── product_1_dalle_1.png
    │   ├── product_1_dalle_2.png
    │   └── product_1_generation_results.json
    ├── product_2/
    └── product_3/
```

## Running Individual Steps

You can also run individual steps:

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

## Troubleshooting

### Common Issues

1. **"No products found in JSON file"**
   - Check that `product_metadata.json` exists and has valid JSON
   - Verify the `products` array is not empty

2. **"No reviews found for product"**
   - Check that the CSV file exists
   - Verify product name in JSON matches product name in CSV
   - Check CSV has `Product` and `Review_Text` columns

3. **"OpenAI API key not found"**
   - Set `OPENAI_API_KEY` in `.env` file
   - Or pass it directly to the orchestrator

4. **"No analysis files found"**
   - Ensure Q1 completed successfully
   - Check that `data/*_data.json` files exist

## Workflow Report

After completion, check `results/workflow_report.json` for:
- Overall workflow status
- Step-by-step completion status
- Generated files
- Error messages
- Duration metrics

## Best Practices

1. **Run Q1 first** to validate your data before proceeding
2. **Check the workflow report** after each run
3. **Use skip flags** to re-run specific steps without redoing everything
4. **Monitor API costs** - Q2 and Q3 make API calls
5. **Keep backups** of your product metadata and CSV files

