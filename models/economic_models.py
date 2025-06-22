import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EconomicModel:
    def __init__(self):
        self.model_params = {}
        self.historical_data = None
        
    def fit(self, data: pd.DataFrame):
        self.historical_data = data
        return self
        
    def predict_gdp(self, years: int = 20, scenario: str = "business_as_usual") -> pd.DataFrame:
        if self.historical_data is None:
            raise ValueError("Model must be fitted before prediction")
            
        current_year = datetime.now().year
        future_years = range(current_year + 1, current_year + years + 1)
        
        predictions = []
        for year in future_years:
            years_elapsed = year - current_year
            
            if scenario == "business_as_usual":
                growth_rate = 0.025
            elif scenario == "moderate_climate_impact":
                growth_rate = 0.02
            elif scenario == "severe_climate_impact":
                growth_rate = 0.015
            else:
                growth_rate = 0.025
                
            gdp = self.historical_data['gdp_billions_usd'].iloc[-1] * (1 + growth_rate) ** years_elapsed
            
            predictions.append({
                'year': year,
                'gdp_billions_usd': gdp,
                'scenario': scenario
            })
            
        return pd.DataFrame(predictions)

class CostBenefitModel:
    def __init__(self):
        self.discount_rate = 0.03
        self.analysis_period = 20
        
    def calculate_net_present_value(self, costs: List[float], benefits: List[float]) -> float:
        if len(costs) != len(benefits):
            raise ValueError("Costs and benefits must have the same length")
            
        npv = 0
        for i, (cost, benefit) in enumerate(zip(costs, benefits)):
            discount_factor = 1 / (1 + self.discount_rate) ** i
            npv += (benefit - cost) * discount_factor
            
        return npv
    
    def calculate_benefit_cost_ratio(self, costs: List[float], benefits: List[float]) -> float:
        total_pv_costs = sum([cost / (1 + self.discount_rate) ** i for i, cost in enumerate(costs)])
        total_pv_benefits = sum([benefit / (1 + self.discount_rate) ** i for i, benefit in enumerate(benefits)])
        
        return total_pv_benefits / total_pv_costs if total_pv_costs > 0 else float('inf')
    
    def analyze_policy_impacts(self, policy_costs: Dict[str, float], 
                             policy_benefits: Dict[str, float]) -> Dict[str, Any]:
        total_costs = sum(policy_costs.values())
        total_benefits = sum(policy_benefits.values())
        
        analysis = {
            'total_costs': total_costs,
            'total_benefits': total_benefits,
            'net_benefits': total_benefits - total_costs,
            'benefit_cost_ratio': total_benefits / total_costs if total_costs > 0 else float('inf'),
            'cost_effectiveness': total_costs / total_benefits if total_benefits > 0 else float('inf')
        }
        
        return analysis

class SectoralEconomicModel:
    def __init__(self):
        self.sector_weights = {}
        self.sector_growth_rates = {}
        
    def set_sector_weights(self, weights: Dict[str, float]):
        self.sector_weights = weights
        
    def set_sector_growth_rates(self, growth_rates: Dict[str, float]):
        self.sector_growth_rates = growth_rates
        
    def forecast_sectoral_gdp(self, base_gdp: float, years: int = 20) -> pd.DataFrame:
        if not self.sector_weights or not self.sector_growth_rates:
            raise ValueError("Sector weights and growth rates must be set")
            
        current_year = datetime.now().year
        future_years = range(current_year + 1, current_year + years + 1)
        
        sectoral_forecasts = []
        for year in future_years:
            years_elapsed = year - current_year
            
            for sector, weight in self.sector_weights.items():
                growth_rate = self.sector_growth_rates.get(sector, 0.02)
                sector_gdp = base_gdp * weight * (1 + growth_rate) ** years_elapsed
                
                sectoral_forecasts.append({
                    'year': year,
                    'sector': sector,
                    'gdp_billions_usd': sector_gdp,
                    'sector_weight': weight
                })
                
        return pd.DataFrame(sectoral_forecasts)
    
    def calculate_sectoral_impacts(self, policy_impacts: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        sectoral_impacts = {}
        
        for sector, impacts in policy_impacts.items():
            if sector in self.sector_weights:
                weight = self.sector_weights[sector]
                sectoral_impacts[sector] = {
                    'gdp_impact_billions': impacts.get('gdp_change', 0) * weight,
                    'employment_impact': impacts.get('employment_change', 0) * weight,
                    'productivity_impact': impacts.get('productivity_change', 0) * weight
                }
                
        return sectoral_impacts

class PolicyEconomicModel:
    def __init__(self):
        self.policy_costs = {}
        self.policy_benefits = {}
        
    def add_policy_cost(self, policy_name: str, cost_structure: Dict[str, float]):
        self.policy_costs[policy_name] = cost_structure
        
    def add_policy_benefit(self, policy_name: str, benefit_structure: Dict[str, float]):
        self.policy_benefits[policy_name] = benefit_structure
        
    def calculate_policy_economic_impact(self, policy_name: str, 
                                       implementation_year: int, 
                                       analysis_period: int = 20) -> Dict[str, Any]:
        if policy_name not in self.policy_costs or policy_name not in self.policy_benefits:
            raise ValueError(f"Policy {policy_name} not found in cost/benefit structures")
            
        costs = self.policy_costs[policy_name]
        benefits = self.policy_benefits[policy_name]
        
        total_costs = sum(costs.values())
        total_benefits = sum(benefits.values())
        
        annual_costs = [total_costs] * analysis_period
        annual_benefits = [total_benefits] * analysis_period
        
        cost_benefit_model = CostBenefitModel()
        npv = cost_benefit_model.calculate_net_present_value(annual_costs, annual_benefits)
        bcr = cost_benefit_model.calculate_benefit_cost_ratio(annual_costs, annual_benefits)
        
        return {
            'policy_name': policy_name,
            'total_costs': total_costs,
            'total_benefits': total_benefits,
            'net_present_value': npv,
            'benefit_cost_ratio': bcr,
            'payback_period': total_costs / total_benefits if total_benefits > 0 else float('inf')
        }
    
    def compare_policies(self, policy_names: List[str]) -> pd.DataFrame:
        comparisons = []
        
        for policy_name in policy_names:
            impact = self.calculate_policy_economic_impact(policy_name, 2024)
            comparisons.append(impact)
            
        return pd.DataFrame(comparisons)

class EmploymentModel:
    def __init__(self):
        self.sector_employment_ratios = {}
        self.job_creation_factors = {}
        
    def set_employment_ratios(self, ratios: Dict[str, float]):
        self.sector_employment_ratios = ratios
        
    def set_job_creation_factors(self, factors: Dict[str, float]):
        self.job_creation_factors = factors
        
    def estimate_job_creation(self, investment_amount: float, sector: str) -> int:
        if sector not in self.job_creation_factors:
            return 0
            
        jobs_per_million = self.job_creation_factors[sector]
        return int(investment_amount * jobs_per_million / 1000000)
    
    def calculate_employment_impact(self, gdp_changes: Dict[str, float]) -> Dict[str, int]:
        employment_impacts = {}
        
        for sector, gdp_change in gdp_changes.items():
            if sector in self.sector_employment_ratios:
                ratio = self.sector_employment_ratios[sector]
                employment_impacts[sector] = int(gdp_change * ratio)
                
        return employment_impacts
