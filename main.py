import sys
import os
import argparse
from pathlib import Path
import pandas as pd
from config.settings import SETTINGS
from workflows.policy_optimization_workflow import build_policy_optimization_graph
from agents.communications_agent import CommunicationsAgent

def get_available_regions(data_path):
    """
    Returns a sorted list of unique regions available in the climate data CSV file located at the given data_path.

    Args:
        data_path (str or Path): Path to the directory containing 'climate_data.csv'.

    Returns:
        list: Sorted list of unique region names found in the climate data file. Returns an empty list if the file is not found.
    """
    climate_file = Path(data_path) / "climate_data.csv"
    if not climate_file.exists():
        print(f"Could not find {climate_file}")
        return []
    df = pd.read_csv(climate_file)
    return sorted(df['region'].unique())

def main():
    """
    Main entry point for the Eco-Policy-Copilot policy optimization workflow.
    Parses command-line arguments, loads configuration, prompts for missing inputs,
    runs the policy optimization workflow, and outputs results including a policy brief and executive summary.
    """
    parser = argparse.ArgumentParser(description="Eco-Policy-Copilot: Policy Optimization Workflow")
    parser.add_argument('--region', type=str, help='Region to analyze (e.g., "North America")')
    parser.add_argument('--target', type=float, help='Target emission reduction percent (default from config)')
    parser.add_argument('--budget', type=float, help='Budget constraint in millions USD (optional)')
    parser.add_argument('--max_iter', type=int, help='Maximum optimization iterations (default from config)')
    args = parser.parse_args()

    data_dir = SETTINGS['data_agent']['data_dir']
    default_target = SETTINGS['policy_generator_agent']['default_target_emission_reduction']
    default_budget = SETTINGS['policy_generator_agent']['default_budget_constraint']
    max_iterations = SETTINGS['workflow']['max_iterations']

    regions = get_available_regions(data_dir)
    if not regions:
        print("No regions found in climate data. Exiting.")
        sys.exit(1)

    region = args.region
    if not region or region not in regions:
        print("Available regions:")
        for r in regions:
            print(f"  - {r}")
        region = input(f"Enter region to analyze [{regions[0]}]: ").strip() or regions[0]
        if region not in regions:
            print(f"Invalid region '{region}'. Exiting.")
            sys.exit(1)

    target = args.target if args.target is not None else default_target
    if target is None:
        target = float(input(f"Enter target emission reduction percent [{default_target}]: ") or default_target)

    budget = args.budget if args.budget is not None else default_budget
    if budget is None:
        budget_input = input("Enter budget constraint in millions USD (or leave blank for no constraint): ").strip()
        budget = float(budget_input) if budget_input else None

    max_iter = args.max_iter if args.max_iter is not None else max_iterations

    print(f"\nRunning policy optimization for region: {region}")
    print(f"Target emission reduction: {target}%")
    print(f"Budget constraint: {budget if budget is not None else 'None'}")
    print(f"Max iterations: {max_iter}\n")

    workflow, initial_state = build_policy_optimization_graph(
        region=region,
        target_emission_reduction=target,
        budget_constraint=budget,
        max_iterations=max_iter
    )
    final_state = workflow.invoke(initial_state)

    output_dir = SETTINGS['communications_agent']['output_dir']
    Path(output_dir).mkdir(exist_ok=True)
    if final_state.get("final_output"):
        print("\n=== Policy Brief ===")
        import json
        print(json.dumps(final_state["final_output"], indent=2, default=str))
        out_file = Path(output_dir) / f"policy_brief_{region.replace(' ', '_').lower()}.json"
        with open(out_file, 'w') as f:
            json.dump(final_state["final_output"], f, indent=2, default=str)
        print(f"\nPolicy brief saved to {out_file}")

        comms_agent = CommunicationsAgent()
        summary = comms_agent.generate_executive_summary(region, policy_recommendations=final_state["final_output"].get('policies'))
        print("\n=== Executive Summary ===")
        print(summary)
        summary_file = Path(output_dir) / f"executive_summary_{region.replace(' ', '_').lower()}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(summary, indent=2, default=str))
        print(f"\nExecutive summary saved to {summary_file}")
    else:
        print("No final output generated.")

if __name__ == "__main__":
    main() 