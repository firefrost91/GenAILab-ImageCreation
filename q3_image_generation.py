"""
Q3: Image Generation with Diffusion Model

This module generates product images using multiple image generation models
based on the analysis from Q2.
"""

import json
import os
import requests
from typing import List, Dict, Any
from openai import OpenAI
from PIL import Image
import time


class ImageGenerator:
    """
    Generates product images using multiple models:
    1. OpenAI DALL-E 3
    2. Stability AI Stable Diffusion
    """
    
    def __init__(self, openai_api_key: str, stability_api_key: str = None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.stability_api_key = stability_api_key
        self.stability_base_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    def craft_image_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Craft an effective prompt for image generation based on LLM analysis.
        Combines information from both prompt engineering and RAG analyses.
        """
        pe_analysis = analysis.get("prompt_engineering", {}).get("analysis", {})
        rag_analysis = analysis.get("rag", {}).get("analysis", {})
        combined = analysis.get("combined_guidance", {})
        
        # Extract key information
        product_summary = pe_analysis.get("summary", "")
        visual_chars = pe_analysis.get("visual_characteristics", {})
        key_features = pe_analysis.get("key_features", [])
        image_guidance = pe_analysis.get("image_generation_guidance", "")
        
        # Build comprehensive prompt
        prompt_parts = []
        
        # Product type and main description
        if product_summary:
            prompt_parts.append(f"Product: {product_summary}")
        
        # Visual characteristics
        if isinstance(visual_chars, dict):
            if "colors" in visual_chars:
                prompt_parts.append(f"Colors: {', '.join(visual_chars['colors'])}")
            if "design_style" in visual_chars:
                prompt_parts.append(f"Design style: {visual_chars['design_style']}")
            if "materials" in visual_chars:
                prompt_parts.append(f"Materials: {', '.join(visual_chars['materials'])}")
        elif isinstance(visual_chars, str):
            prompt_parts.append(f"Visual: {visual_chars}")
        
        # Key features to show
        if key_features:
            features_str = ", ".join(key_features[:5])  # Top 5 features
            prompt_parts.append(f"Key features: {features_str}")
        
        # Image generation guidance
        if image_guidance:
            if isinstance(image_guidance, dict):
                if "visual_style" in image_guidance:
                    prompt_parts.append(f"Style: {image_guidance['visual_style']}")
                if "key_elements" in image_guidance:
                    prompt_parts.append(f"Elements: {', '.join(image_guidance['key_elements'])}")
            elif isinstance(image_guidance, str):
                prompt_parts.append(image_guidance)
        
        # Add professional product photography context
        prompt_parts.append("Professional product photography, clean white background, high quality, detailed, sharp focus")
        
        final_prompt = ". ".join(prompt_parts)
        return final_prompt
    
    def generate_with_dalle(self, prompt: str, model: str = "dall-e-3", 
                           size: str = "1024x1024", quality: str = "standard") -> Dict[str, Any]:
        """
        Generate image using OpenAI DALL-E 3
        """
        try:
            response = self.openai_client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt if hasattr(response.data[0], 'revised_prompt') else prompt
            
            return {
                "success": True,
                "model": "dall-e-3",
                "url": image_url,
                "revised_prompt": revised_prompt,
                "original_prompt": prompt
            }
        except Exception as e:
            return {
                "success": False,
                "model": "dall-e-3",
                "error": str(e),
                "prompt": prompt
            }
    
    def generate_with_stable_diffusion(self, prompt: str, 
                                      style_preset: str = "photographic") -> Dict[str, Any]:
        """
        Generate image using Stability AI Stable Diffusion
        """
        if not self.stability_api_key:
            return {
                "success": False,
                "model": "stable-diffusion",
                "error": "Stability API key not provided"
            }
        
        try:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.stability_api_key}"
            }
            
            body = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    }
                ],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
                "style_preset": style_preset
            }
            
            response = requests.post(
                self.stability_base_url,
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if "artifacts" in data and len(data["artifacts"]) > 0:
                    image_base64 = data["artifacts"][0]["base64"]
                    return {
                        "success": True,
                        "model": "stable-diffusion",
                        "base64": image_base64,
                        "prompt": prompt
                    }
                else:
                    return {
                        "success": False,
                        "model": "stable-diffusion",
                        "error": "No artifacts in response",
                        "response": data
                    }
            else:
                return {
                    "success": False,
                    "model": "stable-diffusion",
                    "error": f"API error: {response.status_code}",
                    "response": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "model": "stable-diffusion",
                "error": str(e),
                "prompt": prompt
            }
    
    def download_image(self, url: str, save_path: str):
        """Download image from URL"""
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    
    def save_base64_image(self, base64_data: str, save_path: str):
        """Save base64 encoded image"""
        import base64
        image_data = base64.b64decode(base64_data)
        with open(save_path, 'wb') as f:
            f.write(image_data)
    
    def generate_images_for_product(self, product_info: Dict[str, Any], 
                                   analysis: Dict[str, Any],
                                   output_dir: str = "generated_images",
                                   num_images: int = 3) -> Dict[str, Any]:
        """
        Generate multiple images for a product using different models and prompts
        """
        product_id = product_info["id"]
        product_name = product_info["name"]
        
        os.makedirs(output_dir, exist_ok=True)
        product_dir = os.path.join(output_dir, product_id)
        os.makedirs(product_dir, exist_ok=True)
        
        print(f"\nGenerating images for: {product_name}")
        print(f"  Product ID: {product_id}")
        
        # Craft base prompt
        base_prompt = self.craft_image_prompt(analysis)
        print(f"  Base prompt: {base_prompt[:200]}...")
        
        # Generate variations of the prompt
        prompt_variations = self._create_prompt_variations(base_prompt, num_images)
        
        results = {
            "product_id": product_id,
            "product_name": product_name,
            "base_prompt": base_prompt,
            "generated_images": []
        }
        
        # Generate with DALL-E
        print(f"\n  Generating with DALL-E 3...")
        dalle_images = []
        for i, prompt in enumerate(prompt_variations[:num_images]):
            print(f"    Image {i+1}/{num_images}...")
            result = self.generate_with_dalle(prompt)
            
            if result["success"]:
                image_filename = f"{product_id}_dalle_{i+1}.png"
                image_path = os.path.join(product_dir, image_filename)
                
                if self.download_image(result["url"], image_path):
                    dalle_images.append({
                        "model": "dall-e-3",
                        "image_path": image_path,
                        "prompt": result.get("revised_prompt", prompt),
                        "original_prompt": prompt,
                        "index": i+1
                    })
                    print(f"      ✓ Saved to {image_filename}")
                else:
                    print(f"      ✗ Failed to download image")
            else:
                print(f"      ✗ Error: {result.get('error', 'Unknown error')}")
            
            time.sleep(1)  # Rate limiting
        
        results["generated_images"].extend(dalle_images)
        
        # Generate with Stable Diffusion (if API key available)
        if self.stability_api_key:
            print(f"\n  Generating with Stable Diffusion...")
            sd_images = []
            for i, prompt in enumerate(prompt_variations[:num_images]):
                print(f"    Image {i+1}/{num_images}...")
                result = self.generate_with_stable_diffusion(prompt)
                
                if result["success"]:
                    image_filename = f"{product_id}_sd_{i+1}.png"
                    image_path = os.path.join(product_dir, image_filename)
                    
                    self.save_base64_image(result["base64"], image_path)
                    sd_images.append({
                        "model": "stable-diffusion",
                        "image_path": image_path,
                        "prompt": prompt,
                        "index": i+1
                    })
                    print(f"      ✓ Saved to {image_filename}")
                else:
                    print(f"      ✗ Error: {result.get('error', 'Unknown error')}")
                
                time.sleep(1)  # Rate limiting
            
            results["generated_images"].extend(sd_images)
        else:
            print(f"\n  Skipping Stable Diffusion (API key not provided)")
        
        # Save results metadata
        results_file = os.path.join(product_dir, f"{product_id}_generation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ Generated {len(results['generated_images'])} images")
        print(f"  ✓ Results saved to {product_dir}")
        
        return results
    
    def _create_prompt_variations(self, base_prompt: str, num_variations: int) -> List[str]:
        """Create variations of the prompt for different image generations"""
        variations = [base_prompt]  # Keep original
        
        # Add variations with different emphasis
        if num_variations > 1:
            variations.append(base_prompt + " Close-up view, detailed textures visible")
        
        if num_variations > 2:
            variations.append(base_prompt + " Side angle view, showing depth and dimensions")
        
        if num_variations > 3:
            variations.append(base_prompt + " Studio lighting, professional product shot")
        
        if num_variations > 4:
            variations.append(base_prompt + " Isolated on white background, high resolution")
        
        # Fill remaining with base prompt if needed
        while len(variations) < num_variations:
            variations.append(base_prompt)
        
        return variations[:num_variations]


def generate_all_product_images(results_dir: str = "results",
                                images_dir: str = "generated_images",
                                openai_api_key: str = None,
                                stability_api_key: str = None,
                                num_images_per_product: int = 3):
    """Generate images for all analyzed products"""
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
    
    generator = ImageGenerator(
        openai_api_key=openai_api_key,
        stability_api_key=stability_api_key
    )
    
    print("=" * 60)
    print("Q3: Image Generation with Diffusion Models")
    print("=" * 60)
    
    # Find all analysis files
    analysis_files = [f for f in os.listdir(results_dir) if f.endswith("_analysis.json")]
    
    all_results = []
    
    # Also load product data files to get proper product info
    data_dir = "data"
    
    for analysis_file in analysis_files:
        file_path = os.path.join(results_dir, analysis_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract product ID from filename
        product_id = data.get("product_id", analysis_file.replace("_analysis.json", ""))
        
        # Try to load original product data for proper name
        product_info = {"id": product_id, "name": "Unknown Product"}
        product_data_file = os.path.join(data_dir, f"{product_id}_data.json")
        
        if os.path.exists(product_data_file):
            with open(product_data_file, 'r', encoding='utf-8') as f:
                product_data = json.load(f)
                product_info["name"] = product_data.get("product_info", {}).get("name", "Unknown Product")
        else:
            # Fallback: try to get product name from analysis
            if "prompt_engineering" in data and "analysis" in data["prompt_engineering"]:
                pe_analysis = data["prompt_engineering"]["analysis"]
                if "summary" in pe_analysis:
                    summary = pe_analysis["summary"]
                    product_info["name"] = summary.split(".")[0] if summary else "Unknown Product"
        
        analysis = data
        
        print(f"\nProcessing: {product_info.get('name', 'Unknown')}")
        
        result = generator.generate_images_for_product(
            product_info,
            analysis,
            output_dir=images_dir,
            num_images=num_images_per_product
        )
        
        all_results.append(result)
    
    print("\n" + "=" * 60)
    print("Image generation complete!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    from config import OPENAI_API_KEY, STABILITY_API_KEY
    
    if not OPENAI_API_KEY:
        print("Error: Please set OPENAI_API_KEY in your .env file")
    else:
        generate_all_product_images(
            openai_api_key=OPENAI_API_KEY,
            stability_api_key=STABILITY_API_KEY,
            num_images_per_product=3
        )

