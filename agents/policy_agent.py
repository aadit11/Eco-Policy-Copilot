import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime
import json
from pathlib import Path

from agents.data_agent import DataAgent
from agents.simulation_agent import SimulationAgent
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class PolicyGeneratorAgent:
    def __init__(self, data_agent: DataAgent = None, simulation_agent: SimulationAgent = None):
        """
        Initialize the PolicyGeneratorAgent with optional data and simulation agents.
        Args:
            data_agent (DataAgent, optional): Data agent instance. Defaults to None.
            simulation_agent (SimulationAgent, optional): Simulation agent instance. Defaults to None.
        """
        self.data_agent = data_agent or DataAgent()
        self.simulation_agent = simulation_agent or SimulationAgent(data_agent)
        self.policy_recommendations = {}
        self.policy_history = []
        self.llm_client = LLMClient()
        
    def initialize(self):
        """
        Initialize the data agent if its data cache is empty.
        """
        if not self.data_agent.data_cache:
            self.data_agent.initialize()
    
    def generate_policy_recommendations(self, region: str, target_emission_reduction: float = 30, 
                                      budget_constraint: float = None, policy_count: int = 5) -> List[Dict]:
        """
        Generate a list of policy recommendations for a given region.
        Args:
            region (str): The region for which to generate recommendations.
            target_emission_reduction (float, optional): Target emission reduction percentage. Defaults to 30.
            budget_constraint (float, optional): Maximum allowed budget. Defaults to None.
            policy_count (int, optional): Number of policy sets to recommend. Defaults to 5.
        Returns:
            List[Dict]: List of recommended policy sets.
        """
        self.initialize()
        
        logger.info(f"Generating policy recommendations for {region}")
        
        baseline_data = self.data_agent.get_climate_data(region)
        policy_data = self.data_agent.get_policy_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        
        if baseline_data.empty or policy_data.empty:
            return []
        
        current_emissions = baseline_data.groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not baseline_data.empty else 0
        target_emissions = current_emissions * (1 - target_emission_reduction / 100)
        
        policy_candidates = self._generate_policy_candidates(region, policy_data, economic_data)
        policy_combinations = self._generate_policy_combinations(policy_candidates, policy_count)
        
        recommendations = []
        
        for i, combination in enumerate(policy_combinations):
            simulation_result = self.simulation_agent.run_policy_simulation(region, combination)
            
            if simulation_result:
                summary = simulation_result['summary']
                final_emissions = summary['final_year_emissions']
                emission_reduction = ((current_emissions - final_emissions) / current_emissions) * 100
                
                if emission_reduction >= target_emission_reduction:
                    if budget_constraint is None or summary['cumulative_policy_cost'] <= budget_constraint:
                        recommendation = {
                            'id': f"policy_set_{i+1}",
                            'policies': combination,
                            'emission_reduction_percent': emission_reduction,
                            'total_cost_millions_usd': summary['cumulative_policy_cost'],
                            'net_benefit_millions_usd': summary['net_policy_impact'],
                            'cost_effectiveness': summary['total_emissions_reduction'] / summary['cumulative_policy_cost'] if summary['cumulative_policy_cost'] > 0 else 0,
                            'feasibility_score': self._calculate_feasibility_score(combination, region),
                            'implementation_timeline': self._estimate_implementation_timeline(combination),
                            'risk_assessment': self._assess_policy_risks(combination, region),
                            'generated_at': datetime.now().isoformat()
                        }
                        recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x['cost_effectiveness'], reverse=True)
        
        self.policy_recommendations[region] = recommendations[:policy_count]
        self.policy_history.extend(recommendations)
        
        return self.policy_recommendations[region]
    
    def _generate_policy_candidates(self, region: str, policy_data: pd.DataFrame, economic_data: pd.DataFrame) -> List[Dict]:
        """
        Generate a list of candidate policies for a region based on available data.
        Args:
            region (str): The region for which to generate candidates.
            policy_data (pd.DataFrame): DataFrame of policy data.
            economic_data (pd.DataFrame): DataFrame of economic data.
        Returns:
            List[Dict]: List of candidate policy dictionaries.
        """
        candidates = []
        
        current_gdp = economic_data.groupby('region', observed=False)['gdp_billions_usd'].last().iloc[0] if not economic_data.empty else 0
        current_emissions = self.data_agent.get_climate_data(region).groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not self.data_agent.get_climate_data(region).empty else 0
        
        carbon_tax_candidates = [
            {
                'name': 'Moderate Carbon Tax',
                'type': 'carbon_tax',
                'tax_rate_usd_per_ton': 30,
                'implementation_delay_years': 1,
                'description': 'Moderate carbon pricing to incentivize emission reductions'
            },
            {
                'name': 'Aggressive Carbon Tax',
                'type': 'carbon_tax',
                'tax_rate_usd_per_ton': 60,
                'implementation_delay_years': 1,
                'description': 'Higher carbon pricing for significant emission reductions'
            }
        ]
        
        renewable_subsidy_candidates = [
            {
                'name': 'Solar Energy Subsidy',
                'type': 'renewable_subsidy',
                'subsidy_rate_percent': 25,
                'implementation_delay_years': 1,
                'description': 'Subsidize solar energy adoption'
            },
            {
                'name': 'Wind Energy Subsidy',
                'type': 'renewable_subsidy',
                'subsidy_rate_percent': 30,
                'implementation_delay_years': 1,
                'description': 'Subsidize wind energy development'
            }
        ]
        
        energy_efficiency_candidates = [
            {
                'name': 'Building Efficiency Standards',
                'type': 'energy_efficiency',
                'efficiency_improvement_percent': 20,
                'implementation_delay_years': 2,
                'description': 'Mandatory building energy efficiency improvements'
            },
            {
                'name': 'Industrial Efficiency Program',
                'type': 'energy_efficiency',
                'efficiency_improvement_percent': 15,
                'implementation_delay_years': 2,
                'description': 'Industrial energy efficiency incentives'
            }
        ]
        
        ev_subsidy_candidates = [
            {
                'name': 'EV Purchase Incentive',
                'type': 'ev_subsidy',
                'ev_penetration_target_percent': 25,
                'implementation_delay_years': 1,
                'description': 'Electric vehicle purchase subsidies'
            },
            {
                'name': 'EV Infrastructure Investment',
                'type': 'ev_subsidy',
                'ev_penetration_target_percent': 15,
                'implementation_delay_years': 2,
                'description': 'Electric vehicle charging infrastructure'
            }
        ]
        
        candidates.extend(carbon_tax_candidates)
        candidates.extend(renewable_subsidy_candidates)
        candidates.extend(energy_efficiency_candidates)
        candidates.extend(ev_subsidy_candidates)
        
        return candidates
    
    def _generate_policy_combinations(self, candidates: List[Dict], max_policies: int) -> List[List[Dict]]:
        """
        Generate valid combinations of policy candidates.
        Args:
            candidates (List[Dict]): List of candidate policies.
            max_policies (int): Maximum number of policies in a combination.
        Returns:
            List[List[Dict]]: List of valid policy combinations.
        """
        combinations = []
        
        for i in range(1, min(max_policies + 1, len(candidates) + 1)):
            for combo in self._get_combinations(candidates, i):
                if self._validate_policy_combination(combo):
                    combinations.append(combo)
        
        return combinations[:50]
    
    def _get_combinations(self, items: List, r: int) -> List[List]:
        """
        Recursively generate all combinations of items of length r.
        Args:
            items (List): List of items to combine.
            r (int): Length of each combination.
        Returns:
            List[List]: List of combinations.
        """
        if r == 0:
            return [[]]
        if r > len(items):
            return []
        
        combinations = []
        for i in range(len(items) - r + 1):
            for combo in self._get_combinations(items[i+1:], r-1):
                combinations.append([items[i]] + combo)
        
        return combinations
    
    def _validate_policy_combination(self, policies: List[Dict]) -> bool:
        """
        Validate a combination of policies based on type constraints.
        Args:
            policies (List[Dict]): List of policy dictionaries.
        Returns:
            bool: True if the combination is valid, False otherwise.
        """
        policy_types = [p['type'] for p in policies]
        
        if policy_types.count('carbon_tax') > 2:
            return False
        
        if policy_types.count('renewable_subsidy') > 2:
            return False
        
        return True
    
    def _calculate_feasibility_score(self, policies: List[Dict], region: str) -> float:
        """
        Calculate the average feasibility score for a set of policies in a region.
        Args:
            policies (List[Dict]): List of policy dictionaries.
            region (str): The region for which to calculate feasibility.
        Returns:
            float: Average feasibility score.
        """
        policy_data = self.data_agent.get_policy_data(region)
        
        if policy_data.empty:
            return 0.5
        
        feasibility_scores = []
        
        for policy in policies:
            policy_type = policy['type']
            regional_policies = policy_data[policy_data['policy_type'] == policy_type]
            
            if not regional_policies.empty:
                avg_feasibility = regional_policies['feasibility_score'].mean()
                feasibility_scores.append(avg_feasibility)
            else:
                feasibility_scores.append(0.5)
        
        return np.mean(feasibility_scores) if feasibility_scores else 0.5
    
    def _estimate_implementation_timeline(self, policies: List[Dict]) -> Dict[str, Any]:
        """
        Estimate the implementation timeline for a set of policies.
        Args:
            policies (List[Dict]): List of policy dictionaries.
        Returns:
            Dict[str, Any]: Timeline details including phases and durations.
        """
        max_delay = max([p.get('implementation_delay_years', 1) for p in policies])
        
        timeline = {
            'total_duration_years': max_delay + 5,
            'phases': [
                {
                    'phase': 'Planning',
                    'duration_months': 6,
                    'activities': ['Policy design', 'Stakeholder consultation', 'Legal framework']
                },
                {
                    'phase': 'Implementation',
                    'duration_months': max_delay * 12,
                    'activities': ['Infrastructure development', 'Program rollout', 'Monitoring setup']
                },
                {
                    'phase': 'Full Operation',
                    'duration_months': 60,
                    'activities': ['Policy enforcement', 'Performance monitoring', 'Adjustments']
                }
            ]
        }
        
        return timeline
    
    def _assess_policy_risks(self, policies: List[Dict], region: str) -> Dict[str, Any]:
        """
        Assess the risks associated with a set of policies in a region.
        Args:
            policies (List[Dict]): List of policy dictionaries.
            region (str): The region for which to assess risks.
        Returns:
            Dict[str, Any]: Risk assessment and mitigation strategies.
        """
        risks = {
            'political_risk': 'medium',
            'economic_risk': 'medium',
            'implementation_risk': 'medium',
            'public_acceptance_risk': 'medium',
            'mitigation_strategies': []
        }
        
        policy_types = [p['type'] for p in policies]
        
        if 'carbon_tax' in policy_types:
            risks['political_risk'] = 'high'
            risks['public_acceptance_risk'] = 'high'
            risks['mitigation_strategies'].append('Gradual implementation with clear communication')
        
        if 'renewable_subsidy' in policy_types:
            risks['economic_risk'] = 'medium'
            risks['mitigation_strategies'].append('Budget caps and sunset clauses')
        
        if 'energy_efficiency' in policy_types:
            risks['implementation_risk'] = 'medium'
            risks['mitigation_strategies'].append('Technical assistance and compliance support')
        
        if 'ev_subsidy' in policy_types:
            risks['economic_risk'] = 'medium'
            risks['mitigation_strategies'].append('Phased rollout with infrastructure planning')
        
        return risks
    
    def optimize_policy_parameters(self, region: str, policy_template: Dict, 
                                 target_emission_reduction: float) -> Dict:
        """
        Optimize the parameters of a policy template to achieve a target emission reduction.
        Args:
            region (str): The region for optimization.
            policy_template (Dict): The policy template to optimize.
            target_emission_reduction (float): Target emission reduction percentage.
        Returns:
            Dict: Optimized policy parameters.
        """
        self.initialize()
        
        logger.info(f"Optimizing parameters for policy: {policy_template['name']}")
        
        best_policy = policy_template.copy()
        best_score = 0
        
        if policy_template['type'] == 'carbon_tax':
            tax_rates = [20, 30, 40, 50, 60, 70, 80]
            for rate in tax_rates:
                test_policy = policy_template.copy()
                test_policy['tax_rate_usd_per_ton'] = rate
                
                result = self.simulation_agent.run_policy_simulation(region, [test_policy])
                if result:
                    summary = result['summary']
                    current_emissions = self.data_agent.get_climate_data(region).groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not self.data_agent.get_climate_data(region).empty else 0
                    emission_reduction = ((current_emissions - summary['final_year_emissions']) / current_emissions) * 100
                    
                    if emission_reduction >= target_emission_reduction:
                        score = summary['net_policy_impact'] / summary['cumulative_policy_cost'] if summary['cumulative_policy_cost'] > 0 else 0
                        if score > best_score:
                            best_score = score
                            best_policy = test_policy
        
        elif policy_template['type'] == 'renewable_subsidy':
            subsidy_rates = [15, 20, 25, 30, 35, 40]
            for rate in subsidy_rates:
                test_policy = policy_template.copy()
                test_policy['subsidy_rate_percent'] = rate
                
                result = self.simulation_agent.run_policy_simulation(region, [test_policy])
                if result:
                    summary = result['summary']
                    current_emissions = self.data_agent.get_climate_data(region).groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not self.data_agent.get_climate_data(region).empty else 0
                    emission_reduction = ((current_emissions - summary['final_year_emissions']) / current_emissions) * 100
                    
                    if emission_reduction >= target_emission_reduction:
                        score = summary['net_policy_impact'] / summary['cumulative_policy_cost'] if summary['cumulative_policy_cost'] > 0 else 0
                        if score > best_score:
                            best_score = score
                            best_policy = test_policy
        
        return best_policy
    
    def get_policy_insights(self, region: str) -> Dict[str, Any]:
        """
        Generate insights about policies for a given region, including effectiveness, cost, feasibility, gaps, and opportunities.
        Args:
            region (str): The region for which to generate insights.
        Returns:
            Dict[str, Any]: Dictionary of policy insights.
        """
        self.initialize()
        
        policy_data = self.data_agent.get_policy_data(region)
        effectiveness_data = self.data_agent.get_policy_effectiveness_data(region)
        
        insights = {
            'most_effective_policy_types': effectiveness_data.nlargest(3, 'co2_reduction_mt_per_year').index.tolist(),
            'most_cost_effective_policy_types': effectiveness_data.nlargest(3, 'cost_effectiveness').index.tolist(),
            'highest_feasibility_policy_types': effectiveness_data.nlargest(3, 'feasibility_score').index.tolist(),
            'policy_gaps': self._identify_policy_gaps(region),
            'regional_opportunities': self._identify_regional_opportunities(region),
            'implementation_lessons': self._extract_implementation_lessons(region)
        }
        
        return insights
    
    def _identify_policy_gaps(self, region: str) -> List[str]:
        """
        Identify gaps in policy coverage for a region.
        Args:
            region (str): The region to analyze.
        Returns:
            List[str]: List of identified policy gaps.
        """
        policy_data = self.data_agent.get_policy_data(region)
        climate_data = self.data_agent.get_climate_data(region)
        
        gaps = []
        
        if policy_data.empty:
            return ['No policy data available for gap analysis']
        
        sector_emissions = climate_data.groupby('sector', observed=False)['co2_emissions_mt'].sum() if not climate_data.empty else pd.Series()
        policy_coverage = policy_data.groupby('policy_type', observed=False).size() if not policy_data.empty else pd.Series()
        
        if 'transportation' in sector_emissions.index and sector_emissions['transportation'] > 0:
            if 'transportation' not in policy_coverage.index:
                gaps.append('Transportation sector policies needed')
        
        if 'industrial' in sector_emissions.index and sector_emissions['industrial'] > 0:
            if 'industrial' not in policy_coverage.index:
                gaps.append('Industrial sector policies needed')
        
        return gaps
    
    def _identify_regional_opportunities(self, region: str) -> List[str]:
        """
        Identify regional opportunities for policy improvement or expansion.
        Args:
            region (str): The region to analyze.
        Returns:
            List[str]: List of regional opportunities.
        """
        energy_data = self.data_agent.get_energy_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        
        opportunities = []
        
        if not energy_data.empty:
            renewable_share = energy_data['renewable_share'].iloc[-1]
            if renewable_share < 0.3:
                opportunities.append('High potential for renewable energy expansion')
        
        if not economic_data.empty:
            gdp_growth = economic_data['gdp_growth_rate'].iloc[-1]
            if gdp_growth > 0.03:
                opportunities.append('Strong economic growth supports green investment')
        
        return opportunities
    
    def _extract_implementation_lessons(self, region: str) -> List[str]:
        """
        Extract lessons learned from policy implementation in a region.
        Args:
            region (str): The region to analyze.
        Returns:
            List[str]: List of implementation lessons.
        """
        policy_data = self.data_agent.get_policy_data(region)
        
        if policy_data.empty:
            return ['No implementation data available']
        
        lessons = []
        
        high_success_policies = policy_data[policy_data['success_rate_percent'] > 80]
        if not high_success_policies.empty:
            lessons.append('High success rates associated with strong stakeholder engagement')
        
        low_cost_policies = policy_data[policy_data['implementation_cost_millions_usd'] < 10]
        if not low_cost_policies.empty:
            lessons.append('Low-cost policies often have high public acceptance')
        
        return lessons
    
    def export_policy_recommendations(self, output_dir: str = "policy_recommendations"):
        """
        Export the current policy recommendations to JSON files in the specified output directory.
        Args:
            output_dir (str, optional): Directory to save recommendations. Defaults to "policy_recommendations".
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for region, recommendations in self.policy_recommendations.items():
            filename = f"{region}_policy_recommendations_{timestamp}.json"
            with open(output_path / filename, 'w') as f:
                json.dump(recommendations, f, indent=2)
        
        logger.info(f"Policy recommendations exported to {output_path}")
    
    def get_policy_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about generated policy recommendations.
        Returns:
            Dict[str, Any]: Metadata including total recommendations, regions analyzed, types, and scores.
        """
        metadata = {
            'total_recommendations': len(self.policy_history),
            'regions_analyzed': list(self.policy_recommendations.keys()),
            'policy_types_generated': list(set([p['type'] for rec in self.policy_history for p in rec['policies']])),
            'last_generation': max([rec['generated_at'] for rec in self.policy_history], default=None),
            'average_feasibility_score': np.mean([rec['feasibility_score'] for rec in self.policy_history]) if self.policy_history else 0
        }
        
        return metadata

    def get_llm_policy_recommendations(self, region: str, target_emission_reduction: float) -> str:
        """
        Use an LLM to generate policy recommendations for a region and target emission reduction.
        Args:
            region (str): The region for which to generate recommendations.
            target_emission_reduction (float): Target emission reduction percentage.
        Returns:
            str: LLM-generated policy recommendations or error message.
        """
        baseline_data = self.data_agent.get_climate_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        energy_data = self.data_agent.get_energy_data(region)
        
        if baseline_data.empty or economic_data.empty:
            return "No data available for policy analysis"
        
        try:
            current_emissions = baseline_data.groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1]
        except Exception as e:
            logger.error(f"Error accessing current_emissions: {e}")
            current_emissions = 0
        try:
            current_gdp = economic_data.groupby('region', observed=False)['gdp_billions_usd'].last().iloc[0]
        except Exception as e:
            logger.error(f"Error accessing current_gdp: {e}")
            current_gdp = 0
        renewable_share = energy_data['renewable_share'].iloc[-1] if not energy_data.empty else 0
        
        prompt = f"""
        Generate climate policy recommendations for {region} to achieve {target_emission_reduction}% emission reduction:
        
        Current Situation:
        - Emissions: {current_emissions:.1f} million tons CO2/year
        - GDP: ${current_gdp:.1f} billion
        - Renewable energy share: {renewable_share:.1%}
        
        Requirements:
        - Target emission reduction: {target_emission_reduction}%
        - Consider economic feasibility
        - Prioritize cost-effective measures
        - Include implementation timeline
        
        Provide 5 specific policy recommendations with:
        1. Policy type and description
        2. Expected emission reduction
        3. Estimated cost
        4. Implementation timeline
        5. Expected benefits
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM policy generation failed: {e}")
            return "LLM analysis unavailable"
    
    def get_llm_policy_optimization(self, region: str, current_policies: List[Dict]) -> str:
        """
        Use an LLM to optimize a set of current policies for a region.
        Args:
            region (str): The region for which to optimize policies.
            current_policies (List[Dict]): List of current policy dictionaries.
        Returns:
            str: LLM-generated optimization recommendations or error message.
        """
        if not current_policies:
            return "No policies to optimize"
        
        policy_summary = []
        for i, policy in enumerate(current_policies[:3]):  # Top 3 policies
            policy_summary.append(f"Policy {i+1}: {policy.get('name', 'Unknown')} - "
                                f"Emission reduction: {policy.get('emission_reduction_percent', 0):.1f}%, "
                                f"Cost: ${policy.get('total_cost_millions_usd', 0):.1f}M")
        
        prompt = f"""
        Optimize the following climate policies for {region}:
        
        Current Policies:
        {chr(10).join(policy_summary)}
        
        Provide optimization recommendations for:
        1. Policy parameter adjustments (tax rates, subsidy levels, etc.)
        2. Implementation sequencing
        3. Complementary policies to enhance effectiveness
        4. Risk mitigation strategies
        5. Cost optimization opportunities
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM policy optimization failed: {e}")
            return "LLM analysis unavailable"
