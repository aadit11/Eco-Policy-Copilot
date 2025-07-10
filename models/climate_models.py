import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta



class ClimateModel:
    """
    A model for simulating climate-related projections such as emissions, temperature rise, and climate impacts.
    """
    def __init__(self):
        """
        Initialize the ClimateModel with empty parameters and no historical data.
        """
        self.model_params = {}
        self.historical_data = None
        
    def fit(self, data: pd.DataFrame):
        """
        Fit the model using historical climate data.
        Args:
            data (pd.DataFrame): Historical climate data.
        Returns:
            self: The fitted model instance.
        """
        self.historical_data = data
        return self
        
    def predict_emissions(self, years: int = 20, scenario: str = "business_as_usual") -> pd.DataFrame:
        """
        Predict future CO2 emissions for a given number of years and scenario.
        Args:
            years (int): Number of years to predict into the future.
            scenario (str): Emissions scenario ('business_as_usual', 'moderate_reduction', 'aggressive_reduction').
        Returns:
            pd.DataFrame: DataFrame containing year, emissions, and scenario.
        Raises:
            ValueError: If the model has not been fitted with historical data.
        """
        if self.historical_data is None:
            raise ValueError("Model must be fitted before prediction")
            
        current_year = datetime.now().year
        future_years = range(current_year + 1, current_year + years + 1)
        
        predictions = []
        for year in future_years:
            years_elapsed = year - current_year
            
            if scenario == "business_as_usual":
                growth_rate = 0.02
            elif scenario == "moderate_reduction":
                growth_rate = 0.01
            elif scenario == "aggressive_reduction":
                growth_rate = -0.01
            else:
                growth_rate = 0.02
                
            emissions = self.historical_data['co2_emissions_mt'].iloc[-1] * (1 + growth_rate) ** years_elapsed
            
            predictions.append({
                'year': year,
                'emissions_mt': emissions,
                'scenario': scenario
            })
            
        return pd.DataFrame(predictions)
    
    def predict_temperature(self, emissions_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict global temperature rise based on emissions data.
        Args:
            emissions_data (pd.DataFrame): DataFrame with yearly emissions.
        Returns:
            pd.DataFrame: DataFrame with year, predicted temperature, and temperature rise.
        """
        base_temp = 14.0
        temp_sensitivity = 0.006
        
        temperature_data = []
        for _, row in emissions_data.iterrows():
            cumulative_emissions = emissions_data[emissions_data['year'] <= row['year']]['emissions_mt'].sum()
            temp_rise = cumulative_emissions * temp_sensitivity / 1000
            temperature_data.append({
                'year': row['year'],
                'temperature_c': base_temp + temp_rise,
                'temperature_rise_c': temp_rise
            })
            
        return pd.DataFrame(temperature_data)
    
    def assess_climate_impacts(self, temperature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess climate impacts based on temperature rise data.
        Args:
            temperature_data (pd.DataFrame): DataFrame with temperature rise information.
        Returns:
            Dict[str, Any]: Dictionary of estimated climate impacts.
        """
        max_temp_rise = temperature_data['temperature_rise_c'].max()
        
        impacts = {
            'sea_level_rise_m': max_temp_rise * 0.2,
            'extreme_weather_frequency': max_temp_rise * 0.15,
            'biodiversity_loss_percent': max_temp_rise * 2.5,
            'agricultural_yield_change_percent': -max_temp_rise * 1.8,
            'health_impacts_percent': max_temp_rise * 3.2
        }
        
        return impacts

class EmissionForecastModel:
    """
    A model for forecasting regional emissions and calculating emission budgets.
    """
    def __init__(self):
        """
        Initialize the EmissionForecastModel with empty parameters and sector weights.
        """
        self.regional_params = {}
        self.sector_weights = {}
        
    def set_regional_params(self, region: str, params: Dict[str, float]):
        """
        Set parameters for a specific region.
        Args:
            region (str): Region name.
            params (Dict[str, float]): Parameters for the region.
        """
        self.regional_params[region] = params
        
    def set_sector_weights(self, weights: Dict[str, float]):
        """
        Set weights for different economic sectors.
        Args:
            weights (Dict[str, float]): Sector weights.
        """
        self.sector_weights = weights
        
    def forecast_regional_emissions(self, region: str, base_year: int, target_year: int, 
                                  policy_impacts: Dict[str, float] = None) -> pd.DataFrame:
        """
        Forecast emissions for a region from base_year to target_year, optionally applying policy impacts.
        Args:
            region (str): Region name.
            base_year (int): Starting year for the forecast.
            target_year (int): Ending year for the forecast.
            policy_impacts (Dict[str, float], optional): Policy impacts by type.
        Returns:
            pd.DataFrame: DataFrame with yearly emissions and cumulative emissions.
        Raises:
            ValueError: If no parameters are set for the region.
        """
        if region not in self.regional_params:
            raise ValueError(f"No parameters set for region: {region}")
            
        params = self.regional_params[region]
        base_emissions = params.get('base_emissions', 100)
        growth_rate = params.get('growth_rate', 0.02)
        
        forecast_data = []
        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            
            emissions = base_emissions * (1 + growth_rate) ** years_elapsed
            
            if policy_impacts:
                for policy_type, reduction in policy_impacts.items():
                    if year >= params.get('policy_start_year', base_year):
                        emissions *= (1 - reduction)
            
            forecast_data.append({
                'year': year,
                'region': region,
                'emissions_mt': emissions,
                'cumulative_emissions_mt': sum([d['emissions_mt'] for d in forecast_data]) + emissions
            })
            
        return pd.DataFrame(forecast_data)
    
    def calculate_emission_budget(self, region: str, target_reduction: float, 
                                base_year: int, target_year: int) -> Dict[str, float]:
        """
        Calculate the emission budget for a region given a target reduction.
        Args:
            region (str): Region name.
            target_reduction (float): Target reduction percentage.
            base_year (int): Starting year.
            target_year (int): Ending year.
        Returns:
            Dict[str, float]: Emission budget and required reduction information.
        """
        params = self.regional_params[region]
        base_emissions = params.get('base_emissions', 100)
        
        business_as_usual = base_emissions * (target_year - base_year)
        target_emissions = business_as_usual * (1 - target_reduction / 100)
        
        return {
            'business_as_usual_mt': business_as_usual,
            'target_emissions_mt': target_emissions,
            'required_reduction_mt': business_as_usual - target_emissions,
            'annual_reduction_rate': (target_reduction / 100) / (target_year - base_year)
        }

class ClimateImpactModel:
    """
    A model for calculating climate-related impacts and adaptation costs.
    """
    def __init__(self):
        """
        Initialize the ClimateImpactModel with an empty dictionary of impact functions.
        """
        self.impact_functions = {}
        
    def add_impact_function(self, impact_type: str, function):
        """
        Add a custom impact function for a specific impact type.
        Args:
            impact_type (str): The type of impact (e.g., 'health', 'economic').
            function (callable): Function to calculate the impact.
        """
        self.impact_functions[impact_type] = function
        
    def calculate_impacts(self, emissions_data: pd.DataFrame, region: str) -> Dict[str, Any]:
        """
        Calculate impacts for a region using the registered impact functions.
        Args:
            emissions_data (pd.DataFrame): DataFrame with emissions data.
            region (str): Region name.
        Returns:
            Dict[str, Any]: Calculated impacts for the region.
        """
        impacts = {}
        
        total_emissions = emissions_data['emissions_mt'].sum()
        peak_emissions = emissions_data['emissions_mt'].max()
        
        if 'health' in self.impact_functions:
            impacts['health_costs_millions'] = self.impact_functions['health'](total_emissions)
            
        if 'economic' in self.impact_functions:
            impacts['economic_damage_millions'] = self.impact_functions['economic'](peak_emissions)
            
        if 'environmental' in self.impact_functions:
            impacts['environmental_damage'] = self.impact_functions['environmental'](total_emissions)
            
        return impacts
    
    def estimate_adaptation_costs(self, temperature_rise: float, region: str) -> Dict[str, float]:
        """
        Estimate adaptation costs for a region based on temperature rise.
        Args:
            temperature_rise (float): Projected temperature rise in degrees Celsius.
            region (str): Region name.
        Returns:
            Dict[str, float]: Estimated adaptation costs by sector (in millions).
        """
        base_costs = {
            'infrastructure': 50,
            'agriculture': 30,
            'healthcare': 20,
            'coastal_protection': 40
        }
        
        adaptation_costs = {}
        for sector, base_cost in base_costs.items():
            adaptation_costs[f'{sector}_cost_millions'] = base_cost * temperature_rise
        
        return adaptation_costs
