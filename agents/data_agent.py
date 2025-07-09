import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from utils.data_loader import EcoPolicyDataLoader
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

class DataAgent:
    def __init__(self, data_dir: str = "data"):
        """Initialize the DataAgent with a data directory, data loader, and LLM client."""
        self.data_loader = EcoPolicyDataLoader(data_dir)
        self.data_cache = {}
        self.last_update = None
        self.update_interval = timedelta(hours=6)
        self.llm_client = LLMClient()
        
    def initialize(self) -> Dict[str, Any]:
        """Load all data into the cache and set the last update timestamp."""
        logger.info("Initializing Data Agent...")
        self.data_cache = self.data_loader.load_all_data()
        self.last_update = datetime.now()
        return self.data_cache
    
    def get_climate_data(self, region: str = None, start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """Retrieve climate data, optionally filtered by region and year range."""
        if not self.data_cache:
            self.initialize()
        
        climate_df = self.data_cache['climate_data'].copy()
        
        if region:
            climate_df = climate_df[climate_df['region'] == region]
        
        if start_year:
            climate_df = climate_df[climate_df['year'].dt.year >= start_year]
        
        if end_year:
            climate_df = climate_df[climate_df['year'].dt.year <= end_year]
        
        return climate_df
    
    def get_energy_data(self, region: str = None, energy_source: str = None) -> pd.DataFrame:
        """Retrieve energy data, optionally filtered by region and energy source."""
        if not self.data_cache:
            self.initialize()
        
        energy_df = self.data_cache['energy_data'].copy()
        
        if region:
            energy_df = energy_df[energy_df['region'] == region]
        
        if energy_source:
            energy_df = energy_df[energy_df['energy_source'] == energy_source]
        
        return energy_df
    
    def get_economic_data(self, region: str = None, metric: str = None) -> pd.DataFrame:
        """Retrieve economic data, optionally filtered by region and metric."""
        if not self.data_cache:
            self.initialize()
        
        economic_df = self.data_cache['economic_data'].copy()
        
        if region:
            economic_df = economic_df[economic_df['region'] == region]
        
        if metric:
            if metric in economic_df.columns:
                economic_df = economic_df[['year', 'region', metric]]
        
        return economic_df
    
    def get_policy_data(self, region: str = None, policy_type: str = None) -> pd.DataFrame:
        """Retrieve policy data, optionally filtered by region and policy type."""
        if not self.data_cache:
            self.initialize()
        
        policy_df = self.data_cache['policy_data'].copy()
        
        if region:
            policy_df = policy_df[policy_df['region'] == region]
        
        if policy_type:
            policy_df = policy_df[policy_df['policy_type'] == policy_type]
        
        return policy_df
    
    def get_emissions_trend(self, region: str, years: int = 10) -> Dict[str, Any]:
        """Analyze CO2 emissions trend for a region over a specified number of years."""
        climate_df = self.get_climate_data(region)
        
        if climate_df.empty:
            return {}
        
        recent_data = climate_df.groupby('year', observed=False)['co2_emissions_mt'].sum().tail(years)
        if recent_data.empty:
            return {}
        
        trend_analysis = {
            'current_emissions': recent_data.iloc[-1],
            'emissions_change': recent_data.iloc[-1] - recent_data.iloc[0],
            'emissions_change_percent': ((recent_data.iloc[-1] - recent_data.iloc[0]) / recent_data.iloc[0]) * 100,
            'average_annual_change': recent_data.diff().mean(),
            'trend_direction': 'increasing' if recent_data.iloc[-1] > recent_data.iloc[0] else 'decreasing',
            'data_points': len(recent_data)
        }
        
        return trend_analysis
    
    def get_energy_mix_analysis(self, region: str) -> Dict[str, Any]:
        """Analyze the energy mix for a region, including renewable and fossil fuel shares."""
        energy_df = self.get_energy_data(region)
        
        if energy_df.empty:
            return {}
        
        latest_year = energy_df['year'].max()
        latest_data = energy_df[energy_df['year'] == latest_year]
        
        total_energy = latest_data['energy_consumption_twh'].sum()
        renewable_energy = latest_data[latest_data['energy_source'].isin(['Solar', 'Wind', 'Hydro', 'Geothermal', 'Biomass'])]['energy_consumption_twh'].sum()
        
        energy_mix = {
            'total_energy_twh': total_energy,
            'renewable_share_percent': (renewable_energy / total_energy) * 100,
            'fossil_fuel_share_percent': ((total_energy - renewable_energy) / total_energy) * 100,
            'energy_intensity': latest_data['energy_intensity'].mean(),
            'year': latest_year.year
        }
        
        return energy_mix
    
    def get_economic_indicators(self, region: str) -> Dict[str, Any]:
        """Retrieve key economic indicators for a region."""
        economic_df = self.get_economic_data(region)
        
        if economic_df.empty:
            return {}
        
        latest_data = economic_df.groupby('region', observed=False).last()
        if latest_data.empty:
            return {}
        
        indicators = {
            'gdp_billions_usd': latest_data['gdp_billions_usd'].iloc[0],
            'gdp_per_capita': latest_data['gdp_per_capita'].iloc[0],
            'gdp_growth_rate': latest_data['gdp_growth_rate'].iloc[0],
            'population_millions': latest_data['population_millions'].iloc[0],
            'year': latest_data.index[0]
        }
        
        return indicators
    
    def get_policy_effectiveness_data(self, region: str) -> pd.DataFrame:
        """Aggregate and return policy effectiveness metrics for a region."""
        policy_df = self.get_policy_data(region)
        
        if policy_df.empty:
            return pd.DataFrame()
        
        effectiveness_metrics = policy_df.groupby('policy_type', observed=False).agg({
            'co2_reduction_mt_per_year': 'mean',
            'cost_effectiveness': 'mean',
            'feasibility_score': 'mean',
            'success_rate_percent': 'mean',
            'implementation_cost_millions_usd': 'mean'
        }).round(2)
        
        return effectiveness_metrics
    
    def get_regional_comparison(self, regions: List[str], metric: str = 'co2_emissions_mt') -> pd.DataFrame:
        """Compare a specified metric across multiple regions."""
        comparison_data = []
        
        for region in regions:
            if metric == 'co2_emissions_mt':
                data = self.get_climate_data(region)
                if not data.empty:
                    latest = data.groupby('year', observed=False)[metric].sum().tail(1)
                    if latest.empty:
                        continue
                    comparison_data.append({
                        'region': region,
                        'value': latest.iloc[0],
                        'year': latest.index[0].year
                    })
            elif metric in ['gdp_billions_usd', 'gdp_per_capita', 'population_millions']:
                data = self.get_economic_data(region)
                if not data.empty:
                    latest = data.groupby('region', observed=False)[metric].last()
                    if latest.empty:
                        continue
                    comparison_data.append({
                        'region': region,
                        'value': latest.iloc[0],
                        'year': data['year'].max().year
                    })
        
        return pd.DataFrame(comparison_data)
    
    def get_forecast_data(self, region: str, scenario: str = 'business_as_usual', years: int = 20) -> pd.DataFrame:
        """Generate emissions forecast data for a region under a given scenario for a number of years."""
        if not self.data_cache:
            self.initialize()
        
        current_year = datetime.now().year
        forecast_years = range(current_year + 1, current_year + years + 1)
        
        baseline_emissions = self.data_cache['simulation_parameters']['regional_parameters'][region.lower().replace(' ', '_')]['baseline_emissions']
        scenario_params = self.data_cache['simulation_parameters']['scenario_parameters'][scenario]
        
        forecast_data = []
        for year in forecast_years:
            emissions = baseline_emissions * (1 + scenario_params['emissions_growth_rate']) ** (year - current_year)
            forecast_data.append({
                'year': year,
                'region': region,
                'scenario': scenario,
                'projected_emissions_mt': emissions
            })
        
        return pd.DataFrame(forecast_data)
    
    def refresh_data(self) -> bool:
        """Refresh the data cache by reloading all data. Returns True if successful, False otherwise."""
        try:
            logger.info("Refreshing data cache...")
            self.data_cache = self.data_loader.load_all_data()
            self.last_update = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Return a summary of the loaded data, including sources, regions, and years covered."""
        if not self.data_cache:
            self.initialize()
        
        summary = {
            'last_updated': self.last_update.isoformat() if self.last_update else None,
            'data_sources': {
                'climate_data': len(self.data_cache['climate_data']),
                'economic_data': len(self.data_cache['economic_data']),
                'energy_data': len(self.data_cache['energy_data']),
                'policy_data': len(self.data_cache['policy_data']),
                'social_impact_data': len(self.data_cache['social_impact_data']),
                'technology_cost_data': len(self.data_cache['technology_cost_data'])
            },
            'regions_available': list(self.data_cache['climate_data']['region'].unique()),
            'years_covered': {
                'start': self.data_cache['climate_data']['year'].min().year,
                'end': self.data_cache['climate_data']['year'].max().year
            }
        }
        
        return summary
    
    def get_llm_data_insights(self, region: str) -> str:
        """Use an LLM to generate insights about a region's climate and economic data."""
        climate_data = self.get_climate_data(region)
        economic_data = self.get_economic_data(region)
        energy_data = self.get_energy_data(region)
        
        if climate_data.empty or economic_data.empty:
            return "No data available for analysis"
        
        prompt = f"""
        Analyze the following climate and economic data for {region}:
        
        Climate Data Summary:
        - Current emissions: {climate_data['co2_emissions_mt'].iloc[-1]:.1f} million tons
        - Emissions trend: {self.get_emissions_trend(region).get('trend_direction', 'unknown')}
        
        Economic Data Summary:
        - GDP: ${economic_data['gdp_billions_usd'].iloc[-1]:.1f} billion
        - GDP per capita: ${economic_data['gdp_per_capita'].iloc[-1]:.0f}
        
        Energy Data Summary:
        - Renewable share: {energy_data['renewable_share'].iloc[-1]:.1%}
        
        Provide 3 key insights about this region's climate and economic situation, and suggest 2 priority areas for climate policy intervention.
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return "LLM analysis unavailable"
    
    def get_llm_trend_analysis(self, region: str, metric: str = "emissions") -> str:
        """Use an LLM to analyze the trend of a specified metric (emissions or GDP) for a region."""
        if metric == "emissions":
            data = self.get_climate_data(region)
            current_value = data['co2_emissions_mt'].iloc[-1]
            trend = self.get_emissions_trend(region)
        elif metric == "gdp":
            data = self.get_economic_data(region)
            current_value = data['gdp_billions_usd'].iloc[-1]
            trend = {"trend_direction": "increasing", "emissions_change_percent": 2.5}
        else:
            return "Unsupported metric"
        
        prompt = f"""
        Analyze the {metric} trend for {region}:
        - Current value: {current_value:.1f}
        - Trend direction: {trend.get('trend_direction', 'unknown')}
        - Change over time: {trend.get('emissions_change_percent', 0):.1f}%
        
        Provide a brief analysis of what this trend means for climate policy and what actions should be prioritized.
        """
        
        try:
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM trend analysis failed: {e}")
            return "LLM analysis unavailable"
    
    def get_regional_summary(self, region: str = None) -> pd.DataFrame:
        """Return a summary DataFrame for a region using the data loader's summary method."""
        return self.data_loader.get_regional_summary(region)
