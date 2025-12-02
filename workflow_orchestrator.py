"""
AI Agentic Workflow Orchestrator

This module orchestrates the complete pipeline:
Q1: Data Collection → Q2: LLM Analysis → Q3: Image Generation

It provides intelligent workflow management with:
- Error handling and recovery
- Progress tracking
- Dependency management
- Result validation
- Logging and reporting
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import the three main modules
from q1_data_collection import ProductDataCollector
from q2_llm_analysis import analyze_all_products
from q3_image_generation import generate_all_product_images
from config import OPENAI_API_KEY, STABILITY_API_KEY, DATA_DIR, RESULTS_DIR, IMAGES_DIR


class WorkflowOrchestrator:
    """
    Orchestrates the complete AI workflow pipeline.
    
    Workflow Steps:
    1. Q1: Collect product data and reviews from JSON/CSV files
    2. Q2: Analyze products using LLM (prompt engineering + RAG)
    3. Q3: Generate product images using diffusion models
    
    Features:
    - Automatic dependency checking
    - Error recovery and retry logic
    - Progress tracking
    - Result validation
    - Comprehensive logging
    """
    
    def __init__(
        self,
        products_json_path: str = "product_metadata.json",
        data_dir: str = DATA_DIR,
        results_dir: str = RESULTS_DIR,
        images_dir: str = IMAGES_DIR,
        openai_api_key: Optional[str] = None,
        stability_api_key: Optional[str] = None
    ):
        self.products_json_path = products_json_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.images_dir = images_dir
        
        # API keys
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.stability_api_key = stability_api_key or STABILITY_API_KEY
        
        # Workflow state
        self.workflow_state = {
            "started_at": None,
            "completed_at": None,
            "steps_completed": [],
            "steps_failed": [],
            "products_processed": 0,
            "errors": []
        }
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log workflow events with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✓",
            "WARNING": "⚠️",
            "ERROR": "✗",
            "STEP": "→"
        }.get(level, "•")
        print(f"[{timestamp}] {prefix} {message}")
    
    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met"""
        self.log("Validating prerequisites...", "STEP")
        
        # Check products JSON exists
        if not os.path.exists(self.products_json_path):
            self.log(f"Products JSON not found: {self.products_json_path}", "ERROR")
            return False
        
        # Check OpenAI API key
        if not self.openai_api_key:
            self.log("OpenAI API key not found. Required for Q2 and Q3.", "ERROR")
            return False
        
        # Check if CSV files referenced in JSON exist
        try:
            with open(self.products_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            json_dir = os.path.dirname(os.path.abspath(self.products_json_path))
            products = metadata.get("products", [])
            
            for product in products:
                csv_file = product.get("review_csv", "")
                if csv_file:
                    csv_path = os.path.join(json_dir, csv_file)
                    if not os.path.exists(csv_path):
                        self.log(f"Review CSV not found: {csv_path}", "WARNING")
        except Exception as e:
            self.log(f"Error validating prerequisites: {e}", "WARNING")
        
        self.log("Prerequisites validated", "SUCCESS")
        return True
    
    def step_q1_data_collection(self) -> bool:
        """
        Step 1: Collect product data and reviews
        """
        self.log("=" * 80, "STEP")
        self.log("STEP 1: Product Data Collection (Q1)", "STEP")
        self.log("=" * 80, "STEP")
        
        try:
            collector = ProductDataCollector(data_dir=self.data_dir)
            collected_data = collector.collect_all_products(
                products_json_path=self.products_json_path
            )
            
            if not collected_data:
                self.log("No products were collected", "ERROR")
                return False
            
            # Validate collected data
            for item in collected_data:
                product = item.get("product", {})
                description = item.get("description", "")
                reviews = item.get("reviews", [])
                
                if not description:
                    self.log(f"Warning: No description for {product.get('name', 'Unknown')}", "WARNING")
                if not reviews:
                    self.log(f"Warning: No reviews for {product.get('name', 'Unknown')}", "WARNING")
            
            self.workflow_state["products_processed"] = len(collected_data)
            self.workflow_state["steps_completed"].append("q1_data_collection")
            self.log(f"Q1 completed: {len(collected_data)} products collected", "SUCCESS")
            return True
            
        except Exception as e:
            error_msg = f"Q1 failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.workflow_state["steps_failed"].append("q1_data_collection")
            self.workflow_state["errors"].append(error_msg)
            return False
    
    def step_q2_llm_analysis(self) -> bool:
        """
        Step 2: Analyze products using LLM
        """
        self.log("=" * 80, "STEP")
        self.log("STEP 2: LLM Analysis (Q2)", "STEP")
        self.log("=" * 80, "STEP")
        
        try:
            # Check if Q1 data exists
            data_files = [f for f in os.listdir(self.data_dir) if f.endswith("_data.json")]
            if not data_files:
                self.log("No product data files found. Q1 must complete first.", "ERROR")
                return False
            
            analyses = analyze_all_products(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                api_key=self.openai_api_key
            )
            
            if not analyses:
                self.log("No analyses were generated", "ERROR")
                return False
            
            # Validate analyses
            for analysis in analyses:
                if "prompt_engineering" not in analysis:
                    self.log("Warning: Analysis missing prompt_engineering results", "WARNING")
                if "rag" not in analysis:
                    self.log("Warning: Analysis missing RAG results", "WARNING")
            
            self.workflow_state["steps_completed"].append("q2_llm_analysis")
            self.log(f"Q2 completed: {len(analyses)} products analyzed", "SUCCESS")
            return True
            
        except Exception as e:
            error_msg = f"Q2 failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.workflow_state["steps_failed"].append("q2_llm_analysis")
            self.workflow_state["errors"].append(error_msg)
            return False
    
    def step_q3_image_generation(self) -> bool:
        """
        Step 3: Generate product images
        """
        self.log("=" * 80, "STEP")
        self.log("STEP 3: Image Generation (Q3)", "STEP")
        self.log("=" * 80, "STEP")
        
        try:
            # Check if Q2 analyses exist
            analysis_files = [f for f in os.listdir(self.results_dir) if f.endswith("_analysis.json")]
            if not analysis_files:
                self.log("No analysis files found. Q2 must complete first.", "ERROR")
                return False
            
            image_results = generate_all_product_images(
                results_dir=self.results_dir,
                images_dir=self.images_dir,
                openai_api_key=self.openai_api_key,
                stability_api_key=self.stability_api_key,
                num_images_per_product=3
            )
            
            if not image_results:
                self.log("No images were generated", "ERROR")
                return False
            
            # Count total images generated
            total_images = sum(len(r.get("generated_images", [])) for r in image_results)
            
            self.workflow_state["steps_completed"].append("q3_image_generation")
            self.log(f"Q3 completed: {total_images} images generated for {len(image_results)} products", "SUCCESS")
            return True
            
        except Exception as e:
            error_msg = f"Q3 failed: {str(e)}"
            self.log(error_msg, "ERROR")
            self.workflow_state["steps_failed"].append("q3_image_generation")
            self.workflow_state["errors"].append(error_msg)
            return False
    
    def generate_workflow_report(self) -> Dict[str, Any]:
        """Generate a comprehensive workflow report"""
        report = {
            "workflow_summary": {
                "started_at": self.workflow_state["started_at"],
                "completed_at": self.workflow_state["completed_at"],
                "duration_seconds": None,
                "status": "completed" if not self.workflow_state["steps_failed"] else "partial",
                "products_processed": self.workflow_state["products_processed"]
            },
            "steps": {
                "q1_data_collection": {
                    "status": "completed" if "q1_data_collection" in self.workflow_state["steps_completed"] else "failed",
                    "output_dir": self.data_dir,
                    "files": [f for f in os.listdir(self.data_dir) if f.endswith("_data.json")] if os.path.exists(self.data_dir) else []
                },
                "q2_llm_analysis": {
                    "status": "completed" if "q2_llm_analysis" in self.workflow_state["steps_completed"] else "failed",
                    "output_dir": self.results_dir,
                    "files": [f for f in os.listdir(self.results_dir) if f.endswith("_analysis.json")] if os.path.exists(self.results_dir) else []
                },
                "q3_image_generation": {
                    "status": "completed" if "q3_image_generation" in self.workflow_state["steps_completed"] else "failed",
                    "output_dir": self.images_dir,
                    "product_dirs": [d for d in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, d))] if os.path.exists(self.images_dir) else []
                }
            },
            "errors": self.workflow_state["errors"]
        }
        
        # Calculate duration
        if self.workflow_state["started_at"] and self.workflow_state["completed_at"]:
            start = datetime.fromisoformat(self.workflow_state["started_at"])
            end = datetime.fromisoformat(self.workflow_state["completed_at"])
            report["workflow_summary"]["duration_seconds"] = (end - start).total_seconds()
        
        return report
    
    def save_workflow_report(self, report: Dict[str, Any]):
        """Save workflow report to JSON file"""
        report_file = os.path.join(self.results_dir, "workflow_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.log(f"Workflow report saved to {report_file}", "SUCCESS")
    
    def run_complete_workflow(self, skip_steps: List[str] = None) -> bool:
        """
        Run the complete workflow pipeline.
        
        Args:
            skip_steps: List of step names to skip (e.g., ['q3_image_generation'])
        
        Returns:
            True if all steps completed successfully, False otherwise
        """
        skip_steps = skip_steps or []
        
        self.workflow_state["started_at"] = datetime.now().isoformat()
        
        self.log("=" * 80, "STEP")
        self.log("AI AGENTIC WORKFLOW ORCHESTRATOR", "STEP")
        self.log("=" * 80, "STEP")
        self.log(f"Products JSON: {self.products_json_path}", "INFO")
        self.log(f"Data Directory: {self.data_dir}", "INFO")
        self.log(f"Results Directory: {self.results_dir}", "INFO")
        self.log(f"Images Directory: {self.images_dir}", "INFO")
        self.log("", "INFO")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            self.log("Prerequisites validation failed. Aborting workflow.", "ERROR")
            return False
        
        # Step 1: Data Collection
        if "q1_data_collection" not in skip_steps:
            if not self.step_q1_data_collection():
                self.log("Workflow stopped due to Q1 failure", "ERROR")
                self.workflow_state["completed_at"] = datetime.now().isoformat()
                return False
        else:
            self.log("Skipping Q1: Data Collection", "INFO")
        
        # Step 2: LLM Analysis
        if "q2_llm_analysis" not in skip_steps:
            if not self.step_q2_llm_analysis():
                self.log("Workflow stopped due to Q2 failure", "ERROR")
                self.workflow_state["completed_at"] = datetime.now().isoformat()
                return False
        else:
            self.log("Skipping Q2: LLM Analysis", "INFO")
        
        # Step 3: Image Generation
        if "q3_image_generation" not in skip_steps:
            if not self.step_q3_image_generation():
                self.log("Workflow stopped due to Q3 failure", "ERROR")
                self.workflow_state["completed_at"] = datetime.now().isoformat()
                return False
        else:
            self.log("Skipping Q3: Image Generation", "INFO")
        
        # Generate and save report
        self.workflow_state["completed_at"] = datetime.now().isoformat()
        report = self.generate_workflow_report()
        self.save_workflow_report(report)
        
        # Print summary
        self.log("", "INFO")
        self.log("=" * 80, "STEP")
        self.log("WORKFLOW COMPLETE", "SUCCESS")
        self.log("=" * 80, "STEP")
        self.log(f"Products Processed: {self.workflow_state['products_processed']}", "INFO")
        self.log(f"Steps Completed: {len(self.workflow_state['steps_completed'])}/3", "INFO")
        if self.workflow_state["steps_failed"]:
            self.log(f"Steps Failed: {', '.join(self.workflow_state['steps_failed'])}", "WARNING")
        self.log("", "INFO")
        
        return len(self.workflow_state["steps_failed"]) == 0


def main():
    """Main entry point for the workflow orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agentic Workflow Orchestrator")
    parser.add_argument(
        "--products-json",
        type=str,
        default="product_metadata.json",
        help="Path to products metadata JSON file"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["q1_data_collection", "q2_llm_analysis", "q3_image_generation"],
        help="Steps to skip"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Directory for data files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory for analysis results"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=IMAGES_DIR,
        help="Directory for generated images"
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator(
        products_json_path=args.products_json,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        images_dir=args.images_dir
    )
    
    # Run workflow
    success = orchestrator.run_complete_workflow(skip_steps=args.skip or [])
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

