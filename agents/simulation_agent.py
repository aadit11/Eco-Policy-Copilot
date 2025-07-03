import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

from agents.data_agent import DataAgent
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class SimulationAgent:
    def __init__(self, data_agent: DataAgent = None):
        self.data_agent = data_agent or DataAgent()
        self.simulation_results = {}
        self.current_scenario = None
        self.llm_client = LLMClient()
        
    def initialize(self):
        if not self.data_agent.data_cache:
            self.data_agent.initialize()
    
    def run_baseline_simulation(self, region: str, years: int = 20) -> Dict[str, Any]:
        self.initialize()
        
        logger.info(f"Running baseline simulation for {region} over {years} years")
        
        current_year = datetime.now().year
        simulation_years = range(current_year, current_year + years + 1)
        
        baseline_data = self.data_agent.get_climate_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        
        if baseline_data.empty or economic_data.empty:
            return {}
        
        latest_emissions = baseline_data.groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not baseline_data.empty else 0
        latest_gdp = economic_data.groupby('region', observed=False)['gdp_billions_usd'].last().iloc[0] if not economic_data.empty else 0
        latest_population = economic_data.groupby('region', observed=False)['population_millions'].last().iloc[0] if not economic_data.empty else 0
        
        simulation_results = []
        
        for year in simulation_years:
            years_elapsed = year - current_year
            
            emissions = latest_emissions * (1.02) ** years_elapsed
            gdp = latest_gdp * (1.025) ** years_elapsed
            population = latest_population * (1.01) ** years_elapsed
            
            energy_intensity = 0.15 * (0.98) ** years_elapsed
            energy_consumption = gdp * energy_intensity
            
            renewable_share = min(0.25 + (years_elapsed * 0.01), 0.8)
            fossil_energy = energy_consumption * (1 - renewable_share)
            
            temperature_rise = 0.02 * years_elapsed
            
            simulation_results.append({
                'year': year,
                'region': region,
                'emissions_mt': emissions,
                'gdp_billions_usd': gdp,
                'population_millions': population,
                'energy_consumption_twh': energy_consumption,
                'renewable_share': renewable_share,
                'fossil_energy_twh': fossil_energy,
                'temperature_rise_c': temperature_rise,
                'emissions_per_capita': emissions / population,
                'gdp_per_capita': gdp * 1000 / population
            })
        
        baseline_df = pd.DataFrame(simulation_results)
        
        self.simulation_results['baseline'] = {
            'data': baseline_df,
            'summary': self._calculate_simulation_summary(baseline_df),
            'region': region,
            'scenario': 'baseline',
            'timestamp': datetime.now().isoformat()
        }
        
        return self.simulation_results['baseline']
    
    def run_policy_simulation(self, region: str, policies: List[Dict], years: int = 20) -> Dict[str, Any]:
        self.initialize()
        
        logger.info(f"Running policy simulation for {region} with {len(policies)} policies")
        
        baseline_result = self.run_baseline_simulation(region, years)
        if not baseline_result:
            return {}
        
        baseline_df = baseline_result['data'].copy()
        policy_df = baseline_df.copy()
        
        policy_effects = self._calculate_policy_effects(policies, baseline_df)
        
        for year_idx, year in enumerate(policy_df['year']):
            cumulative_emission_reduction = 0
            cumulative_cost = 0
            cumulative_benefits = 0
            
            for policy in policies:
                policy_effect = policy_effects[policy['name']]
                
                if year_idx >= policy_effect['implementation_delay']:
                    years_active = year_idx - policy_effect['implementation_delay'] + 1
                    
                    emission_reduction = policy_effect['annual_emission_reduction'] * min(years_active, policy_effect['ramp_up_years'])
                    cumulative_emission_reduction += emission_reduction
                    
                    cost = policy_effect['annual_cost'] * years_active
                    cumulative_cost += cost
                    
                    benefits = policy_effect['annual_benefits'] * years_active
                    cumulative_benefits += benefits
            
            policy_df.loc[year_idx, 'emissions_mt'] = max(0, baseline_df.loc[year_idx, 'emissions_mt'] - cumulative_emission_reduction)
            policy_df.loc[year_idx, 'policy_cost_millions_usd'] = cumulative_cost
            policy_df.loc[year_idx, 'policy_benefits_millions_usd'] = cumulative_benefits
            policy_df.loc[year_idx, 'net_policy_impact_millions_usd'] = cumulative_benefits - cumulative_cost
        
        policy_df['emissions_reduction_mt'] = baseline_df['emissions_mt'] - policy_df['emissions_mt']
        policy_df['emissions_reduction_percent'] = (policy_df['emissions_reduction_mt'] / baseline_df['emissions_mt']) * 100
        
        policy_result = {
            'data': policy_df,
            'summary': self._calculate_simulation_summary(policy_df),
            'policy_effects': policy_effects,
            'region': region,
            'scenario': 'policy_intervention',
            'policies_applied': policies,
            'timestamp': datetime.now().isoformat()
        }
        
        self.simulation_results['policy'] = policy_result
        return policy_result
    
    def _calculate_policy_effects(self, policies: List[Dict], baseline_df: pd.DataFrame) -> Dict[str, Dict]:
        policy_effects = {}
        
        for policy in policies:
            policy_type = policy.get('type', 'unknown')
            
            if policy_type == 'carbon_tax':
                effect = self._calculate_carbon_tax_effect(policy, baseline_df)
            elif policy_type == 'renewable_subsidy':
                effect = self._calculate_renewable_subsidy_effect(policy, baseline_df)
            elif policy_type == 'energy_efficiency':
                effect = self._calculate_energy_efficiency_effect(policy, baseline_df)
            elif policy_type == 'ev_subsidy':
                effect = self._calculate_ev_subsidy_effect(policy, baseline_df)
            else:
                effect = self._calculate_generic_policy_effect(policy, baseline_df)
            
            policy_effects[policy['name']] = effect
        
        return policy_effects
    
    def _calculate_carbon_tax_effect(self, policy: Dict, baseline_df: pd.DataFrame) -> Dict:
        tax_rate = policy.get('tax_rate_usd_per_ton', 50)
        implementation_delay = policy.get('implementation_delay_years', 1)
        
        baseline_emissions = baseline_df['emissions_mt'].mean()
        price_elasticity = -0.3
        
        annual_emission_reduction = baseline_emissions * abs(price_elasticity) * (tax_rate / 100)
        annual_cost = baseline_emissions * tax_rate * 0.1
        annual_benefits = baseline_emissions * tax_rate * 0.8
        
        return {
            'annual_emission_reduction': annual_emission_reduction,
            'annual_cost': annual_cost,
            'annual_benefits': annual_benefits,
            'implementation_delay': implementation_delay,
            'ramp_up_years': 3
        }
    
    def _calculate_renewable_subsidy_effect(self, policy: Dict, baseline_df: pd.DataFrame) -> Dict:
        subsidy_rate = policy.get('subsidy_rate_percent', 30)
        implementation_delay = policy.get('implementation_delay_years', 1)
        
        baseline_energy = baseline_df['energy_consumption_twh'].mean()
        renewable_potential = baseline_energy * 0.4
        
        annual_emission_reduction = renewable_potential * 0.5 * 0.5
        annual_cost = renewable_potential * 0.1 * subsidy_rate / 100
        annual_benefits = annual_emission_reduction * 50
        
        return {
            'annual_emission_reduction': annual_emission_reduction,
            'annual_cost': annual_cost,
            'annual_benefits': annual_benefits,
            'implementation_delay': implementation_delay,
            'ramp_up_years': 5
        }
    
    def _calculate_energy_efficiency_effect(self, policy: Dict, baseline_df: pd.DataFrame) -> Dict:
        efficiency_improvement = policy.get('efficiency_improvement_percent', 15)
        implementation_delay = policy.get('implementation_delay_years', 2)
        
        baseline_energy = baseline_df['energy_consumption_twh'].mean()
        
        annual_emission_reduction = baseline_energy * (efficiency_improvement / 100) * 0.5
        annual_cost = baseline_energy * 0.05
        annual_benefits = annual_emission_reduction * 50 + baseline_energy * (efficiency_improvement / 100) * 0.08
        
        return {
            'annual_emission_reduction': annual_emission_reduction,
            'annual_cost': annual_cost,
            'annual_benefits': annual_benefits,
            'implementation_delay': implementation_delay,
            'ramp_up_years': 4
        }
    
    def _calculate_ev_subsidy_effect(self, policy: Dict, baseline_df: pd.DataFrame) -> Dict:
        ev_penetration_target = policy.get('ev_penetration_target_percent', 20)
        implementation_delay = policy.get('implementation_delay_years', 1)
        
        baseline_emissions = baseline_df['emissions_mt'].mean()
        transport_share = 0.3
        
        annual_emission_reduction = baseline_emissions * transport_share * (ev_penetration_target / 100) * 0.7
        annual_cost = baseline_emissions * transport_share * (ev_penetration_target / 100) * 0.2
        annual_benefits = annual_emission_reduction * 50
        
        return {
            'annual_emission_reduction': annual_emission_reduction,
            'annual_cost': annual_cost,
            'annual_benefits': annual_benefits,
            'implementation_delay': implementation_delay,
            'ramp_up_years': 6
        }
    
    def _calculate_generic_policy_effect(self, policy: Dict, baseline_df: pd.DataFrame) -> Dict:
        emission_reduction = policy.get('emission_reduction_mt_per_year', 1)
        cost = policy.get('annual_cost_millions_usd', 10)
        benefits = policy.get('annual_benefits_millions_usd', 20)
        implementation_delay = policy.get('implementation_delay_years', 1)
        
        return {
            'annual_emission_reduction': emission_reduction,
            'annual_cost': cost,
            'annual_benefits': benefits,
            'implementation_delay': implementation_delay,
            'ramp_up_years': 3
        }
    
    def _calculate_simulation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {}
        
        summary = {
            'total_emissions_mt': df['emissions_mt'].sum(),
            'average_emissions_mt_per_year': df['emissions_mt'].mean(),
            'emissions_trend': 'decreasing' if df['emissions_mt'].iloc[-1] < df['emissions_mt'].iloc[0] else 'increasing',
            'total_gdp_billions_usd': df['gdp_billions_usd'].sum(),
            'average_gdp_growth': df['gdp_billions_usd'].pct_change(fill_method=None).mean() * 100,
            'cumulative_policy_cost': df.get('policy_cost_millions_usd', pd.Series([0] * len(df))).sum(),
            'cumulative_policy_benefits': df.get('policy_benefits_millions_usd', pd.Series([0] * len(df))).sum(),
            'net_policy_impact': df.get('net_policy_impact_millions_usd', pd.Series([0] * len(df))).sum(),
            'total_emissions_reduction': df.get('emissions_reduction_mt', pd.Series([0] * len(df))).sum(),
            'final_year_emissions': df['emissions_mt'].iloc[-1],
            'final_year_gdp': df['gdp_billions_usd'].iloc[-1]
        }
        
        return summary
    
    def compare_scenarios(self, scenarios: List[str]) -> pd.DataFrame:
        comparison_data = []
        
        for scenario in scenarios:
            if scenario in self.simulation_results:
                result = self.simulation_results[scenario]
                summary = result['summary']
                
                comparison_data.append({
                    'scenario': scenario,
                    'region': result['region'],
                    'total_emissions_mt': summary['total_emissions_mt'],
                    'average_emissions_mt_per_year': summary['average_emissions_mt_per_year'],
                    'total_gdp_billions_usd': summary['total_gdp_billions_usd'],
                    'cumulative_policy_cost': summary['cumulative_policy_cost'],
                    'cumulative_policy_benefits': summary['cumulative_policy_benefits'],
                    'net_policy_impact': summary['net_policy_impact'],
                    'total_emissions_reduction': summary['total_emissions_reduction']
                })
        
        return pd.DataFrame(comparison_data)
    
    def get_sensitivity_analysis(self, region: str, policy: Dict, parameter: str, range_values: List[float]) -> pd.DataFrame:
        sensitivity_results = []
        
        for value in range_values:
            modified_policy = policy.copy()
            modified_policy[parameter] = value
            
            result = self.run_policy_simulation(region, [modified_policy])
            if result:
                summary = result['summary']
                sensitivity_results.append({
                    'parameter': parameter,
                    'value': value,
                    'total_emissions_mt': summary['total_emissions_mt'],
                    'net_policy_impact': summary['net_policy_impact'],
                    'total_emissions_reduction': summary['total_emissions_reduction']
                })
        
        return pd.DataFrame(sensitivity_results)
    
    def export_simulation_results(self, output_dir: str = "simulation_results"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for scenario, result in self.simulation_results.items():
            if 'data' in result:
                filename = f"{scenario}_{timestamp}.csv"
                result['data'].to_csv(output_path / filename, index=False)
                
                summary_filename = f"{scenario}_summary_{timestamp}.json"
                with open(output_path / summary_filename, 'w') as f:
                    json.dump(result['summary'], f, indent=2)
        
        logger.info(f"Simulation results exported to {output_path}")
    
    def get_simulation_metadata(self) -> Dict[str, Any]:
        metadata = {
            'total_scenarios': len(self.simulation_results),
            'scenarios': list(self.simulation_results.keys()),
            'regions_simulated': list(set([r['region'] for r in self.simulation_results.values() if 'region' in r])),
            'last_simulation': max([r['timestamp'] for r in self.simulation_results.values() if 'timestamp' in r], default=None),
            'total_data_points': sum([len(r['data']) for r in self.simulation_results.values() if 'data' in r])
        }
        
        return metadata
    
    def get_llm_simulation_analysis(self, region: str, simulation_result: Dict) -> str:
        if not simulation_result or 'summary' not in simulation_result:
            return "LLM analysis unavailable"
        
        summary = simulation_result['summary']
        
        prompt = f"""
        Analyze the following climate policy simulation results for {region}:
        
        Simulation Summary:
        - Total emissions: {summary.get('total_emissions_mt', 0):.1f} million tons
        - Average annual emissions: {summary.get('average_emissions_mt_per_year', 0):.1f} million tons
        - Total GDP: ${summary.get('total_gdp_billions_usd', 0):.1f} billion
        - Policy cost: ${summary.get('cumulative_policy_cost', 0):.1f} million
        - Net policy impact: ${summary.get('net_policy_impact', 0):.1f} million
        - Total emissions reduction: {summary.get('total_emissions_reduction', 0):.1f} million tons
        
        Provide insights on:
        1. The effectiveness of the simulated policies
        2. Economic implications
        3. Recommendations for policy optimization
        4. Potential risks or concerns
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM simulation analysis failed: {e}")
            return "LLM analysis unavailable"
    
    def get_llm_scenario_comparison(self, scenarios: List[str]) -> str:
        comparison_data = self.compare_scenarios(scenarios)
        
        if comparison_data.empty:
            return "LLM analysis unavailable"
        
        prompt = f"""
        Compare the following climate policy scenarios:
        
        {comparison_data.to_string()}
        
        Provide a comprehensive analysis of:
        1. Which scenario is most effective for emissions reduction
        2. Which scenario offers the best economic return
        3. Trade-offs between different scenarios
        4. Recommendations for policy implementation
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM scenario comparison failed: {e}")
            return "LLM analysis unavailable"
