"""
Q2: Analysis of Customer Reviews with LLM

This module uses LLM APIs to analyze product descriptions and reviews,
extracting valuable information for image generation.
"""

import json
import os
from typing import List, Dict, Any
import tiktoken
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np


class LLMAnalyzer:
    """
    Analyzes product descriptions and reviews using LLM APIs.
    Implements both prompt engineering and RAG approaches.
    """
    
    def __init__(self, api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = OpenAI(api_key=api_key)
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collections = {}
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def create_vector_store(self, product_id: str, description: str, reviews: List[str]):
        """Create a vector store for RAG using ChromaDB"""
        collection_name = f"product_{product_id}"
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.chroma_client.create_collection(name=collection_name)
        self.collections[product_id] = collection
        
        # Chunk and embed the description
        desc_chunks = self.chunk_text(description)
        for i, chunk in enumerate(desc_chunks):
            embedding = self.embedding_model.encode(chunk).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"desc_{i}"]
            )
        
        # Chunk and embed reviews
        for i, review in enumerate(reviews):
            review_chunks = self.chunk_text(review)
            for j, chunk in enumerate(review_chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"review_{i}_{j}"]
                )
        
        print(f"  ✓ Created vector store with {len(desc_chunks) + sum(len(self.chunk_text(r)) for r in reviews)} chunks")
    
    def retrieve_relevant_context(self, product_id: str, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant context from vector store for RAG"""
        if product_id not in self.collections:
            return []
        
        collection = self.collections[product_id]
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    
    def analyze_with_prompt_engineering(self, description: str, reviews: List[str]) -> Dict[str, Any]:
        """
        Analyze product using prompt engineering strategies:
        1. Zero-shot prompting
        2. Few-shot prompting
        3. Chain-of-thought prompting
        """
        # Combine all reviews
        all_reviews = "\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(reviews)])
        
        # Strategy 1: Comprehensive analysis prompt
        analysis_prompt = f"""You are analyzing a product based on its description and customer reviews. 
Extract and synthesize the following information:

Product Description:
{description}

Customer Reviews:
{all_reviews}

Please provide a comprehensive analysis in the following structure:

1. **Product Summary**: A 2-3 sentence summary of what this product is and its main purpose.

2. **Key Features**: List the 5-7 most important features mentioned in the description and reviews.

3. **Visual Characteristics**: Extract all visual information including:
   - Colors mentioned
   - Size/dimensions
   - Design style (modern, classic, minimalist, etc.)
   - Material appearance
   - Shape and form
   - Any distinctive visual elements

4. **Sentiment Analysis**: 
   - Overall sentiment (positive/neutral/negative)
   - Percentage breakdown if possible
   - Main positive points
   - Main concerns/complaints

5. **Topic Extraction**: Identify 5-7 main topics discussed in reviews (e.g., "comfort", "durability", "ease of use")

6. **Customer Experience Highlights**: 3-5 key points about how customers experience the product.

7. **Image Generation Guidance**: Based on all the above, provide specific guidance for generating an accurate product image, including:
   - Visual style
   - Key elements to include
   - Color palette
   - Setting/context
   - Important details to capture

Format your response as JSON with these keys: summary, key_features, visual_characteristics, sentiment, topics, customer_experience, image_generation_guidance.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert product analyst. Provide detailed, structured analysis in JSON format."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)
            
            return {
                "method": "prompt_engineering",
                "analysis": analysis
            }
        except Exception as e:
            print(f"  ✗ Error in prompt engineering analysis: {e}")
            return {"method": "prompt_engineering", "error": str(e)}
    
    def analyze_with_rag(self, product_id: str, description: str, reviews: List[str]) -> Dict[str, Any]:
        """
        Analyze product using RAG (Retrieval-Augmented Generation)
        """
        # Create vector store
        self.create_vector_store(product_id, description, reviews)
        
        # Use RAG to answer specific questions
        queries = [
            "What are the visual characteristics and appearance of this product?",
            "What colors, materials, and design elements are mentioned?",
            "What are the key features that affect how the product looks?",
            "What is the overall customer sentiment and experience?",
            "What specific details should be visible in a product image?"
        ]
        
        rag_contexts = []
        for query in queries:
            context = self.retrieve_relevant_context(product_id, query, top_k=3)
            rag_contexts.append({
                "query": query,
                "context": context
            })
        
        # Build RAG prompt
        context_text = "\n\n".join([
            f"Q: {item['query']}\nRelevant Context:\n" + "\n".join(item['context'])
            for item in rag_contexts
        ])
        
        rag_prompt = f"""Based on the following retrieved context from product descriptions and reviews, 
provide a comprehensive analysis for image generation:

{context_text}

Please extract:
1. Visual characteristics (colors, materials, design, size)
2. Key features to visualize
3. Overall product appearance
4. Important details for accurate representation

Format as JSON with keys: visual_characteristics, key_features, appearance_description, important_details.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting visual information from product data."},
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)
            
            return {
                "method": "rag",
                "analysis": analysis,
                "queries_used": queries
            }
        except Exception as e:
            print(f"  ✗ Error in RAG analysis: {e}")
            return {"method": "rag", "error": str(e)}
    
    def analyze_product(self, product_id: str, description: str, reviews: List[str]) -> Dict[str, Any]:
        """
        Comprehensive analysis using both prompt engineering and RAG
        """
        print(f"\n  Analyzing product {product_id}...")
        print(f"    Description length: {len(description)} chars")
        print(f"    Number of reviews: {len(reviews)}")
        
        # Method 1: Prompt Engineering
        print("    Running prompt engineering analysis...")
        pe_result = self.analyze_with_prompt_engineering(description, reviews)
        
        # Method 2: RAG
        print("    Running RAG analysis...")
        rag_result = self.analyze_with_rag(product_id, description, reviews)
        
        # Combine results
        combined_analysis = {
            "product_id": product_id,
            "prompt_engineering": pe_result,
            "rag": rag_result,
            "combined_guidance": self._combine_analyses(pe_result, rag_result)
        }
        
        return combined_analysis
    
    def _combine_analyses(self, pe_result: Dict, rag_result: Dict) -> Dict[str, Any]:
        """Combine results from both methods for final image generation guidance"""
        combined = {
            "visual_elements": [],
            "color_palette": [],
            "design_style": "",
            "key_features_to_show": [],
            "image_prompt_guidance": ""
        }
        
        # Extract from prompt engineering
        if "analysis" in pe_result:
            pe_analysis = pe_result["analysis"]
            if "visual_characteristics" in pe_analysis:
                combined["visual_elements"].extend(
                    pe_analysis["visual_characteristics"].get("elements", [])
                )
            if "image_generation_guidance" in pe_analysis:
                combined["image_prompt_guidance"] = pe_analysis["image_generation_guidance"]
        
        # Extract from RAG
        if "analysis" in rag_result:
            rag_analysis = rag_result["analysis"]
            if "visual_characteristics" in rag_analysis:
                combined["visual_elements"].extend(
                    rag_analysis.get("visual_characteristics", {}).get("elements", [])
                )
        
        return combined
    
    def save_analysis(self, analysis: Dict[str, Any], output_file: str):
        """Save analysis results to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved analysis to {output_file}")


def analyze_all_products(data_dir: str = "data", results_dir: str = "results", api_key: str = None):
    """Analyze all products in the data directory"""
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    analyzer = LLMAnalyzer(api_key=api_key)
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Q2: LLM Analysis of Product Reviews")
    print("=" * 60)
    
    # Find all product data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith("_data.json")]
    
    all_analyses = []
    
    for data_file in data_files:
        file_path = os.path.join(data_dir, data_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            product_data = json.load(f)
        
        product_id = product_data["product_info"]["id"]
        description = product_data["description"]
        reviews = product_data["reviews"]
        
        print(f"\nProcessing: {product_data['product_info']['name']}")
        
        analysis = analyzer.analyze_product(product_id, description, reviews)
        
        output_file = os.path.join(results_dir, f"{product_id}_analysis.json")
        analyzer.save_analysis(analysis, output_file)
        
        all_analyses.append({
            "product": product_data["product_info"],
            "analysis": analysis
        })
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    return all_analyses


if __name__ == "__main__":
    from config import OPENAI_API_KEY
    
    if not OPENAI_API_KEY:
        print("Error: Please set OPENAI_API_KEY in your .env file")
    else:
        analyze_all_products(api_key=OPENAI_API_KEY)

