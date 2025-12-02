"""
Q1: Product Selection and Customer Review Data Collection

This module handles the collection of product descriptions and customer reviews
from local JSON and CSV files.
"""

import json
import os
import csv
from typing import List, Dict, Any


class ProductDataCollector:
    """
    Collects product data and reviews from local files.
    
    Reads:
    - Product information from JSON file (format: {"products": [{"name": "...", "description": "..."}]})
    - Customer reviews from CSV file (format: Product,Rating,Author,Date,Verified,Review_Title,Review_Text)
    
    Outputs standardized JSON files for q2_llm_analysis.py:
        data/product_<n>_data.json with:
        {
            "product_info": {"id": "...", "name": "...", "category": "...", "rationale": "..."},
            "description": "<text>",
            "reviews": ["...", "..."],
            "num_reviews": <int>
        }
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize product names for matching between files."""
        if not name:
            return ""
        return " ".join(name.strip().lower().split())

    def load_products_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load products from JSON file matching stanley_product.json format.
        
        Expected format:
        {
            "products": [
                {
                    "name": "Product Name",
                    "url": "https://...",
                    "description": "Product description text..."
                }
            ]
        }
        
        Returns list of product dicts with normalized names for matching.
        """
        if not json_path or not os.path.exists(json_path):
            print(f"⚠️  Product JSON not found at {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        products = []
        
        # Handle {"products": [...]} format
        if isinstance(data, dict) and "products" in data:
            for item in data["products"]:
                if isinstance(item, dict) and "name" in item:
                    products.append({
                        "name": item["name"],
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                        "normalized_name": self._normalize_name(item["name"]),
                        "review_csv": item.get("review_csv", "")
                    })
        # Handle direct list format
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    products.append({
                        "name": item["name"],
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                        "normalized_name": self._normalize_name(item["name"])
                    })

        print(f"✓ Loaded {len(products)} products from {json_path}")
        return products

    def load_reviews_from_csv(self, csv_path: str) -> Dict[str, List[str]]:
        """
        Load reviews from CSV file matching stanley_reviews.csv format.
        
        Expected CSV format (comma-separated):
        Product,Rating,Author,Date,Verified,Review_Title,Review_Text
        
        Returns: { normalized_product_name: [review_text_1, review_text_2, ...] }
        """
        if not csv_path or not os.path.exists(csv_path):
            print(f"⚠️  Reviews CSV not found at {csv_path}")
            return {}

        reviews_by_product: Dict[str, List[str]] = {}
        
        # Try different encodings in case the file isn't UTF-8
        encodings = ["utf-8", "iso-8859-1", "latin-1", "cp1252"]
        encoding_used = None
        f = None
        
        for enc in encodings:
            try:
                f = open(csv_path, "r", encoding=enc)
                # Test by reading first line
                f.readline()
                f.seek(0)
                encoding_used = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                if f:
                    f.close()
                    f = None
                continue
        
        if not encoding_used:
            print(f"⚠️  Could not read CSV file with any supported encoding")
            return {}
        
        with open(csv_path, "r", encoding=encoding_used) as f:
            # Read first line to detect delimiter
            first_line = f.readline()
            f.seek(0)
            
            # Check if it's tab or comma separated
            has_tabs = '\t' in first_line and first_line.count('\t') > first_line.count(',')
            delimiter = "\t" if has_tabs else ","
            
            reader = csv.DictReader(f, delimiter=delimiter, quotechar='"')
            fieldnames_raw = reader.fieldnames or []
            
            # Handle BOM and clean fieldnames
            fieldnames = []
            for fn in fieldnames_raw:
                # Remove BOM if present
                fn_clean = fn.strip().strip('\ufeff').strip('"')
                fieldnames.append(fn_clean)
            
            # If we got a single field with commas, split it
            if len(fieldnames) == 1 and ',' in fieldnames[0]:
                fieldnames = [col.strip().strip('"').strip('\ufeff') for col in fieldnames[0].split(',')]
                # Recreate reader with proper fieldnames
                f.seek(0)
                next(f)  # Skip header
                reader = csv.DictReader(f, delimiter=delimiter, quotechar='"', fieldnames=fieldnames)
            
            # Check for required columns (case-insensitive)
            fieldnames_lower = [fn.lower() for fn in fieldnames]
            
            if "product" not in fieldnames_lower:
                print(f"⚠️  'Product' column not found in CSV. Found columns: {fieldnames}")
                return {}
            
            if "review_text" not in fieldnames_lower:
                print(f"⚠️  'Review_Text' column not found in CSV. Found columns: {fieldnames}")
                return {}
            
            # Find the actual column names - try both with and without BOM
            product_col = None
            review_text_col = None
            
            # Try to get first row to see actual keys
            try:
                first_row_pos = f.tell()
                first_row = next(reader, None)
                if first_row:
                    actual_keys = list(first_row.keys())
                    # Try to find columns in actual row keys
                    for key in actual_keys:
                        key_clean = key.strip('\ufeff').strip()
                        if key_clean.lower() == "product" and not product_col:
                            product_col = key
                        elif key_clean.lower() == "review_text" and not review_text_col:
                            review_text_col = key
                    # Reset to beginning
                    f.seek(first_row_pos)
                    reader = csv.DictReader(f, delimiter=delimiter, quotechar='"', fieldnames=fieldnames)
            except:
                pass
            
            # Fallback: use fieldnames if not found in actual keys
            if not product_col or not review_text_col:
                for col in fieldnames:
                    col_clean = col.strip('\ufeff').strip()
                    if col_clean.lower() == "product" and not product_col:
                        product_col = col
                    elif col_clean.lower() == "review_text" and not review_text_col:
                        review_text_col = col
            
            if not product_col or not review_text_col:
                print(f"⚠️  Could not find required columns. Fieldnames: {fieldnames}")
                return {}
            
            # Read reviews - use actual keys from first row
            row_count = 0
            for row in reader:
                row_count += 1
                
                # Try to get product name - check both with and without BOM
                product_name = row.get(product_col) or row.get('\ufeffProduct') or ""
                # Also try any key that contains "product"
                if not product_name:
                    for key in row.keys():
                        if "product" in key.lower():
                            product_name = row.get(key) or ""
                            break
                
                product_name = product_name.strip() if product_name else ""
                review_text = (row.get(review_text_col) or "").strip()
                review_title = (row.get("Review_Title") or row.get("review_title") or "").strip()
                
                # Clean up product name - remove extra quotes and tabs
                product_name = product_name.strip('\t').strip()
                if product_name.startswith('"') and product_name.endswith('"'):
                    product_name = product_name[1:-1].strip()
                if product_name.startswith('"""') and product_name.endswith('"""'):
                    product_name = product_name[3:-3].strip()
                
                # If Review_Text is empty, try combining Review_Title and Review_Text
                # (sometimes they're concatenated in the CSV)
                if not review_text and review_title:
                    # Try to get the last field which might contain both title and text
                    all_values = list(row.values())
                    if len(all_values) >= 7:
                        # Last field might have both title and text concatenated
                        last_field = all_values[-1] or ""
                        if last_field and len(last_field) > len(review_title):
                            review_text = last_field
                
                # Clean up review text - remove extra quotes
                if review_text:
                    if review_text.startswith('"') and review_text.endswith('"'):
                        review_text = review_text[1:-1].strip()
                    if review_text.startswith('"""') and review_text.endswith('"""'):
                        review_text = review_text[3:-3].strip()
                    # Remove any trailing tabs
                    review_text = review_text.strip('\t').strip()
                
                if not product_name or not review_text:
                    continue
                
                normalized_name = self._normalize_name(product_name)
                if normalized_name not in reviews_by_product:
                    reviews_by_product[normalized_name] = []
                reviews_by_product[normalized_name].append(review_text)
            
            print(f"   Processed {row_count} rows from CSV")

        print(f"✓ Loaded reviews for {len(reviews_by_product)} products from {csv_path}")
        return reviews_by_product

    def save_product_data(self, product: Dict[str, Any], description: str, reviews: List[str], product_id: str):
        """Save collected product data to JSON file"""
        output_file = os.path.join(self.data_dir, f"{product_id}_data.json")

        output_data = {
            "product_info": {
                "id": product_id,
                "name": product["name"],
                "category": product.get("category", "Unknown"),
                "rationale": product.get("rationale", ""),
                "url": product.get("url", "")
            },
            "description": description.strip(),
            "reviews": reviews,
            "num_reviews": len(reviews),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved data for {product['name']} to {output_file}")
        return output_file

    def collect_all_products(
        self,
        products_json_path: str,
    ):
        """
        Collect data for all products from JSON file.
        Each product in JSON should have a 'review_csv' field specifying its review CSV file.
        
        Args:
            products_json_path: Path to JSON file with product information (includes review_csv filenames)
        """
        print("=" * 60)
        print("Q1: Product Selection and Data Collection")
        print("=" * 60)
        
        # Load products from JSON
        print(f"\nLoading products from: {products_json_path}")
        products = self.load_products_from_json(products_json_path)
        
        if not products:
            print("⚠️  No products found in JSON file. Exiting.")
            return []
        
        # Load reviews from CSV files specified in product metadata
        print(f"\nLoading reviews from CSV files specified in product metadata...")
        reviews_map = {}  # Maps normalized_product_name -> [reviews]
        for product in products:
            csv_filename = product.get("review_csv", "")
            if not csv_filename:
                print(f"    ⚠️  No review_csv specified for '{product['name']}'")
                continue
            
            # Assume CSV files are in the same directory as the JSON file
            json_dir = os.path.dirname(os.path.abspath(products_json_path))
            csv_path = os.path.join(json_dir, csv_filename)
            
            if not os.path.exists(csv_path):
                print(f"    ⚠️  CSV file not found: {csv_path}")
                continue
            
            print(f"    Loading reviews from: {csv_filename}")
            product_reviews_map = self.load_reviews_from_csv(csv_path)
            
            # Match reviews to this product by normalized name
            product_normalized_name = product["normalized_name"]
            
            # Strategy 1: Exact match
            if product_normalized_name in product_reviews_map:
                reviews_map[product_normalized_name] = product_reviews_map[product_normalized_name]
                print(f"      ✓ Found {len(reviews_map[product_normalized_name])} reviews for '{product['name']}'")
            else:
                # Strategy 2: Partial/fuzzy matching
                found = False
                best_match = None
                best_match_score = 0
                
                # Extract key words from product name (remove common words)
                product_words = set(product_normalized_name.split())
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'in', 'on', 'at', 'to', 'of'}
                product_keywords = product_words - common_words
                
                for csv_normalized_name, csv_reviews in product_reviews_map.items():
                    # Check if normalized names share significant keywords
                    csv_words = set(csv_normalized_name.split())
                    csv_keywords = csv_words - common_words
                    
                    # Calculate match score based on shared keywords
                    shared_keywords = product_keywords & csv_keywords
                    if shared_keywords:
                        match_score = len(shared_keywords) / max(len(product_keywords), len(csv_keywords), 1)
                        
                        # Also check if one contains the other (for partial matches)
                        if product_normalized_name in csv_normalized_name or csv_normalized_name in product_normalized_name:
                            match_score += 0.3
                        
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_match = (csv_normalized_name, csv_reviews)
                
                # Use best match if score is reasonable (at least 30% keyword overlap)
                if best_match and best_match_score >= 0.3:
                    csv_normalized_name, csv_reviews = best_match
                    reviews_map[product_normalized_name] = csv_reviews
                    print(f"      ✓ Found {len(csv_reviews)} reviews for '{product['name']}' (fuzzy matched: '{csv_normalized_name}', score: {best_match_score:.2f})")
                    found = True
                
                if not found:
                    print(f"      ⚠️  No reviews found matching '{product['name']}' (normalized: '{product_normalized_name}')")
                    print(f"      Available normalized names in CSV: {list(product_reviews_map.keys())}")
                    print(f"      Product keywords: {product_keywords}")

        
        print("\n" + "=" * 60)
        print("Processing products...")
        print("=" * 60)
        
        collected_data = []
        
        for idx, product in enumerate(products, 1):
            product_id = f"product_{idx}"
            normalized_name = product["normalized_name"]
            
            print(f"\n[{idx}/{len(products)}] {product['name']}")
            
            # Get description from product data
            description = product.get("description", "")
            if not description:
                print(f"    ⚠️  No description found for this product")
            
            # Get reviews from CSV - reviews_map contains {normalized_name: [reviews]}
            # But we need to get reviews for this specific product's normalized name
            product_reviews = reviews_map.get(normalized_name, [])
            if not product_reviews:
                print(f"    ⚠️  No reviews found for this product (searched for: '{normalized_name}')")
            else:
                print(f"    ✓ Found {len(product_reviews)} reviews")
            
            reviews = product_reviews
            
            # Save product data
            output_file = self.save_product_data(
                product,
                description,
                reviews,
                product_id
            )
            
            collected_data.append({
                "product": product,
                "data_file": output_file,
                "description": description,
                "reviews": reviews
            })
        
        print("\n" + "=" * 60)
        print("Data collection complete!")
        print(f"Processed {len(collected_data)} products")
        print("=" * 60)
        
        return collected_data


if __name__ == "__main__":
    collector = ProductDataCollector()
    
    # Example usage - CSV filenames are now read from product_metadata.json
    collector.collect_all_products(
        products_json_path="product_metadata.json"
    )
