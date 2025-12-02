"""
Main execution script for the Generative AI Lab project.

This script uses the WorkflowOrchestrator to run the complete pipeline:
Q1: Product Selection and Data Collection
Q2: LLM Analysis of Reviews
Q3: Image Generation
"""

from workflow_orchestrator import WorkflowOrchestrator
from config import OPENAI_API_KEY, STABILITY_API_KEY, DATA_DIR, RESULTS_DIR, IMAGES_DIR


def main():
    """Run the complete workflow using the orchestrator"""
    
    # Initialize the workflow orchestrator
    orchestrator = WorkflowOrchestrator(
        products_json_path="product_metadata.json",
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        images_dir=IMAGES_DIR,
        openai_api_key=OPENAI_API_KEY,
        stability_api_key=STABILITY_API_KEY
    )
    
    # Run the complete workflow
    success = orchestrator.run_complete_workflow()
    
    if success:
        print("\n🎉 All workflow steps completed successfully!")
        return 0
    else:
        print("\n❌ Workflow completed with errors. Check the report for details.")
        return 1


if __name__ == "__main__":
    exit(main())
