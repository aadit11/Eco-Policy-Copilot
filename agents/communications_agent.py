import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

from agents.data_agent import DataAgent
from agents.simulation_agent import SimulationAgent
from agents.policy_agent import PolicyGeneratorAgent
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class CommunicationsAgent:
    """
    Agent responsible for generating, summarizing, and exporting climate policy communications.
    Interfaces with data, simulation, and policy agents to produce executive summaries, policy briefs, and related outputs.
    """
    def __init__(self, data_agent: DataAgent = None, simulation_agent: SimulationAgent = None, 
                 policy_agent: PolicyGeneratorAgent = None):
        """
        Initialize the CommunicationsAgent with optional data, simulation, and policy agents.
        Args:
            data_agent (DataAgent, optional): Data agent instance. Defaults to None.
            simulation_agent (SimulationAgent, optional): Simulation agent instance. Defaults to None.
            policy_agent (PolicyGeneratorAgent, optional): Policy agent instance. Defaults to None.
        """
        self.data_agent = data_agent or DataAgent()
        self.simulation_agent = simulation_agent or SimulationAgent(data_agent)
        self.policy_agent = policy_agent or PolicyGeneratorAgent(data_agent, simulation_agent)
        self.communication_history = []
        self.llm_client = LLMClient()
        
    def initialize(self):
        """
        Initialize the data agent if its data cache is empty.
        """
        if not self.data_agent.data_cache:
            self.data_agent.initialize()
    
    def generate_executive_summary(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, Any]:
        self.initialize()
        logger.info(f"Generating LLM executive summary for {region}")
        return self.generate_llm_executive_summary(region, policy_recommendations)

    def generate_policy_brief(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, Any]:
        self.initialize()
        logger.info(f"Generating LLM policy brief for {region}")
        return self.generate_llm_policy_brief(region, policy_recommendations)

    def generate_llm_executive_summary(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, str]:
        if not policy_recommendations:
            policy_recommendations = self.policy_agent.generate_policy_recommendations(region)
        if not policy_recommendations:
            return {"error": "No policy recommendations available"}
        best_policy = policy_recommendations[0]
        baseline_data = self.data_agent.get_climate_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        current_emissions = baseline_data.groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not baseline_data.empty else 0
        current_gdp = economic_data.groupby('region', observed=False)['gdp_billions_usd'].last().iloc[0] if not economic_data.empty else 0
        prompt = f"""
        Create an executive summary for climate policy recommendations in {region}:
        Current Situation:
        - Annual emissions: {current_emissions:.1f} million tons CO2
        - GDP: ${current_gdp:.1f} billion
        Recommended Policy Package:
        - Emission reduction target: {best_policy['emission_reduction_percent']:.1f}%
        - Total cost: ${best_policy['total_cost_millions_usd']:.1f} million
        - Net benefit: ${best_policy['net_benefit_millions_usd']:.1f} million
        - Cost-effectiveness: {best_policy['cost_effectiveness']:.2f} tons CO2/$M
        Write a compelling executive summary that:
        1. Clearly states the problem and urgency
        2. Presents the recommended solution
        3. Highlights key benefits and economic returns
        4. Addresses potential concerns
        5. Provides clear next steps
        6. Uses language appropriate for government officials
        """
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM executive summary generation failed: {e}")
            return "LLM analysis unavailable"

    def generate_llm_policy_brief(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, str]:
        if not policy_recommendations:
            policy_recommendations = self.policy_agent.generate_policy_recommendations(region)
        if not policy_recommendations:
            return {"error": "No policy recommendations available"}
        policy_details = []
        for i, policy in enumerate(policy_recommendations[:3]):
            policy_details.append(f"""
            Policy {i+1}: {policy.get('name', 'Unknown Policy')}
            - Emission reduction: {policy.get('emission_reduction_percent', 0):.1f}%
            - Cost: ${policy.get('total_cost_millions_usd', 0):.1f}M
            - Net benefit: ${policy.get('net_benefit_millions_usd', 0):.1f}M
            - Feasibility score: {policy.get('feasibility_score', 0):.1f}/10
            """)
        prompt = f"""
        Create a comprehensive policy brief for {region} climate action:
        Policy Recommendations:
        {chr(10).join(policy_details)}
        Write a detailed policy brief that includes:
        1. Executive Summary
        2. Background and Context
        3. Policy Analysis and Recommendations
        4. Economic Impact Assessment
        5. Implementation Roadmap
        6. Risk Assessment and Mitigation
        7. Monitoring and Evaluation Framework
        8. Conclusion and Next Steps
        Use professional, government-appropriate language and include specific, actionable recommendations.
        """
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM policy brief generation failed: {e}")
            return "LLM analysis unavailable"
