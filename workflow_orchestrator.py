import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

from openai import OpenAI
from q1_data_collection import ProductDataCollector
from q2_llm_analysis import analyze_all_products
from q3_image_generation import generate_all_product_images

from config import (
    OPENAI_API_KEY,
    STABILITY_API_KEY,
    DATA_DIR,
    RESULTS_DIR,
    IMAGES_DIR
)


# ===========================================================
#                  AGENTIC ORCHESTRATOR
# ===========================================================

class AgenticWorkflow:
    """
    A fully agentic orchestrator:
    - LLM-driven decision making
    - Reflection & self-correction
    - Tool-based pipeline (Q1/Q2/Q3)
    - Autonomous control-loop
    - Persistent memory for improvement
    """

    def __init__(
        self,
        products_json_path: str = "product_metadata.json",
        data_dir: str = DATA_DIR,
        results_dir: str = RESULTS_DIR,
        images_dir: str = IMAGES_DIR
    ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.products_json_path = products_json_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.images_dir = images_dir

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # Agent memory
        self.memory = []
        self.last_action = None

        self.state = {
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "products_processed": 0,
            "steps_run": [],
            "steps_failed": []
        }

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------
    def log(self, msg: str):
        print(f"[AGENT] {msg}")

    # ------------------------------------------------------------
    # Tools (Q1 / Q2 / Q3)
    # ------------------------------------------------------------
    def tool_q1(self):
        """Collect product data."""
        try:
            collector = ProductDataCollector(data_dir=self.data_dir)
            data = collector.collect_all_products(self.products_json_path)

            self.state["products_processed"] = len(data)
            return {"status": "success", "data": data}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def tool_q2(self):
        """Run LLM analysis."""
        try:
            analyses = analyze_all_products(
                data_dir=self.data_dir,
                results_dir=self.results_dir,
                api_key=OPENAI_API_KEY
            )
            return {"status": "success", "analyses": analyses}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def tool_q3(self):
        """Generate images."""
        try:
            imgs = generate_all_product_images(
                results_dir=self.results_dir,
                images_dir=self.images_dir,
                openai_api_key=OPENAI_API_KEY,
                stability_api_key=STABILITY_API_KEY,
                num_images_per_product=3
            )
            return {"status": "success", "images": imgs}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Mapping for the agent to pick tools
    TOOLS = {
        "run_q1": tool_q1,
        "run_q2": tool_q2,
        "run_q3": tool_q3
    }

    # ------------------------------------------------------------
    # Agent Brain: Decide next action
    # ------------------------------------------------------------
    def agent_decide(self) -> Dict[str, Any]:
        prompt = f"""
You are an AI workflow planner.

Your goal:
Process products end-to-end:
1. Collect data (q1)
2. Analyze using LLM (q2)
3. Generate images (q3)

Here is the current workflow state:
{json.dumps(self.state, indent=2)}

Here is your memory:
{json.dumps(self.memory, indent=2)}

Choose the next action from:
- run_q1
- run_q2
- run_q3
- retry_last
- stop

Respond with JSON ONLY:
{{
  "action": "...",
  "reason": "..."
}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.choices[0].message.content)

    # ------------------------------------------------------------
    # Reflection: Evaluate output quality
    # ------------------------------------------------------------
    def reflect(self, action: str, output: Dict[str, Any]) -> bool:
        """Return True if quality is acceptable."""
        reflection_prompt = f"""
You are a reflection agent.

The tool action: {action}
The tool output: {json.dumps(output, indent=2)}

Evaluate the quality.
Return JSON ONLY:
{{
  "quality_score": 0-1,
  "should_retry": true/false,
  "feedback": "..."
}}
"""
        res = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": reflection_prompt}]
        )

        review = json.loads(res.choices[0].message.content)
        self.memory.append({
            "action": action,
            "quality_score": review["quality_score"],
            "feedback": review["feedback"]
        })

        return not review["should_retry"]

    # ------------------------------------------------------------
    # Main autonomous loop
    # ------------------------------------------------------------
    def run(self):
        self.log("Starting agentic workflow...")

        while True:
            decision = self.agent_decide()
            action = decision["action"]
            self.log(f"Agent chose: {action}")

            if action == "stop":
                break

            if action == "retry_last" and self.last_action:
                action = self.last_action

            if action not in self.TOOLS:
                self.log(f"Invalid action: {action}")
                break

            # Execute tool
            tool_fn = self.TOOLS[action]
            output = tool_fn(self)

            self.last_action = action
            self.state["steps_run"].append(action)

            if output["status"] == "error":
                self.state["steps_failed"].append(action)
                self.memory.append({"error": output["error"]})
                continue

            # Reflect & self-correct
            ok = self.reflect(action, output)
            if not ok:
                self.state["steps_failed"].append(action)
                continue

            # Reset failures upon success
            if action in self.state["steps_failed"]:
                self.state["steps_failed"].remove(action)

        self.state["completed_at"] = datetime.now().isoformat()
        self.log("Workflow completed.")

        # Save report
        with open(os.path.join(self.results_dir, "agentic_report.json"), "w") as f:
            json.dump({"state": self.state, "memory": self.memory}, f, indent=2)

        return True


def main():
    agent = AgenticWorkflow()
    agent.run()


if __name__ == "__main__":
    main()
