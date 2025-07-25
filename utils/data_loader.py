import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
import logging
from utils.data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcoPolicyDataLoader:
    """
    Loads, preprocesses, and manages eco-policy related datasets including climate, economic, energy, policy, social impact, and technology cost data.
    Provides methods for data aggregation, policy recommendations, emissions forecasting, and exporting processed data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader with a directory containing data files.
        Args:
            data_dir (str): Path to the data directory.
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        self.simulation_params = None
        self.preprocessor = DataPreprocessor()
        
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all relevant data files and cache them.
        Returns:
            Dict[str, Any]: Dictionary containing all loaded datasets.
        """
        logger.info("Loading all data files...")
        
        data = {
            'climate_data': self.load_climate_data(),
            'economic_data': self.load_economic_data(),
            'energy_data': self.load_energy_data(),
            'policy_data': self.load_policy_data(),
            'social_impact_data': self.load_social_impact_data(),
            'technology_cost_data': self.load_technology_cost_data(),
            'simulation_parameters': self.load_simulation_parameters()
        }
        
        self.data_cache = data
        
        logger.info("All data loaded successfully")
        return data
    
    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing value imputation and feature engineering to a DataFrame.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = self.preprocessor.fill_missing(df)
        df = self.preprocessor.feature_engineering(df)
        return df
    
    def load_climate_data(self) -> pd.DataFrame:
        """
        Load and preprocess climate data from CSV.
        Returns:
            pd.DataFrame: Climate data with engineered features.
        """
        file_path = self.data_dir / "climate_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df['region'] = df['region'].astype('category')
            df['sector'] = df['sector'].astype('category')
            
            df['total_emissions'] = df.groupby(['year', 'region'], observed=False)['co2_emissions_mt'].transform('sum')
            df['emissions_intensity'] = df['co2_emissions_mt'] / df['gdp_billions_usd']
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded climate data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading climate data: {e}")
            return pd.DataFrame()
    
    def load_economic_data(self) -> pd.DataFrame:
        """
        Load and preprocess economic data from CSV.
        Returns:
            pd.DataFrame: Economic data with engineered features.
        """
        file_path = self.data_dir / "economic_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df['region'] = df['region'].astype('category')
            
            df['gdp_per_capita'] = df['gdp_billions_usd'] * 1000 / df['population_millions']
            df['gdp_growth_rate'] = df.groupby('region', observed=False)['gdp_billions_usd'].pct_change(fill_method=None)
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded economic data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading economic data: {e}")
            return pd.DataFrame()
    
    def load_energy_data(self) -> pd.DataFrame:
        """
        Load and preprocess energy data from CSV.
        Returns:
            pd.DataFrame: Energy data with engineered features.
        """
        file_path = self.data_dir / "energy_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df['region'] = df['region'].astype('category')
            df['energy_source'] = df['energy_source'].astype('category')
            
            df['energy_intensity'] = df['energy_consumption_twh'] / df['gdp_billions_usd']
            df['renewable_share'] = df['renewable_energy_twh'] / df['total_energy_twh']
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded energy data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading energy data: {e}")
            return pd.DataFrame()
    
    def load_policy_data(self) -> pd.DataFrame:
        """
        Load and preprocess policy data from CSV.
        Returns:
            pd.DataFrame: Policy data with engineered features.
        """
        file_path = self.data_dir / "policy_database.csv"
        try:
            df = pd.read_csv(file_path)
            df['region'] = df['region'].astype('category')
            df['policy_type'] = df['policy_type'].astype('category')
            
            df['cost_effectiveness'] = df['co2_reduction_mt_per_year'] / df['implementation_cost_millions_usd']
            df['total_cost'] = df['implementation_cost_millions_usd'] + (df['annual_operating_cost_millions_usd'] * 10)
            df['feasibility_score'] = (df['political_feasibility_score'] + df['public_acceptance_score']) / 2
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded policy data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading policy data: {e}")
            return pd.DataFrame()
    
    def load_social_impact_data(self) -> pd.DataFrame:
        """
        Load and preprocess social impact data from CSV.
        Returns:
            pd.DataFrame: Social impact data with engineered features.
        """
        file_path = self.data_dir / "social_impact_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df['region'] = df['region'].astype('category')
            
            df['health_cost_per_capita'] = df['health_costs_millions_usd'] / df['population_millions']
            df['quality_of_life_index'] = (df['life_expectancy'] + df['education_index'] + df['income_index']) / 3
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded social impact data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading social impact data: {e}")
            return pd.DataFrame()
    
    def load_technology_cost_data(self) -> pd.DataFrame:
        """
        Load and preprocess technology cost data from CSV.
        Returns:
            pd.DataFrame: Technology cost data with engineered features.
        """
        file_path = self.data_dir / "technology_cost_data.csv"
        try:
            df = pd.read_csv(file_path)
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            df['technology'] = df['technology'].astype('category')
            df['region'] = df['region'].astype('category')
            
            df['cost_trend'] = df.groupby(['technology', 'region'], observed=False)['cost_per_unit'].pct_change(fill_method=None)
            df['learning_rate'] = -df.groupby(['technology', 'region'], observed=False)['cost_per_unit'].pct_change(fill_method=None) / df.groupby(['technology', 'region'], observed=False)['cumulative_capacity'].pct_change(fill_method=None)
            
            df = self._preprocess_df(df)
            logger.info(f"Loaded technology cost data: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading technology cost data: {e}")
            return pd.DataFrame()
    
    def load_simulation_parameters(self) -> Dict[str, Any]:
        """
        Load simulation parameters from a JSON file.
        Returns:
            Dict[str, Any]: Simulation parameters.
        """
        file_path = self.data_dir / "simulation_parameters.json"
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            self.simulation_params = params
            logger.info("Loaded simulation parameters")
            return params
        except Exception as e:
            logger.error(f"Error loading simulation parameters: {e}")
            return {}
    
    def get_regional_summary(self, region: str = None) -> pd.DataFrame:
        """
        Aggregate and summarize climate and economic data for a given region.
        Args:
            region (str, optional): Region to filter by. If None, summarizes all regions.
        Returns:
            pd.DataFrame: Regional summary statistics.
        """
        if not self.data_cache:
            self.load_all_data()
        
        climate_df = self.data_cache['climate_data']
        economic_df = self.data_cache['economic_data']
        
        if region:
            climate_df = climate_df[climate_df['region'] == region]
            economic_df = economic_df[economic_df['region'] == region]
        
        summary = climate_df.groupby(['region', 'year'], observed=False).agg({
            'co2_emissions_mt': 'sum',
            'temperature_anomaly_c': 'mean',
            'renewable_energy_share': 'mean',
            'population_millions': 'mean',
            'gdp_billions_usd': 'mean'
        }).reset_index()
        
        return summary
    
    def get_policy_recommendations(self, region: str, budget_constraint: float = None) -> pd.DataFrame:
        """
        Generate policy recommendations for a region, optionally filtered by budget constraint.
        Args:
            region (str): Region to generate recommendations for.
            budget_constraint (float, optional): Maximum allowed implementation cost.
        Returns:
            pd.DataFrame: Sorted policy recommendations with composite scores.
        """
        if not self.data_cache:
            self.load_all_data()
        
        policy_df = self.data_cache['policy_data']
        regional_policies = policy_df[policy_df['region'] == region].copy()
        
        if budget_constraint:
            regional_policies = regional_policies[regional_policies['implementation_cost_millions_usd'] <= budget_constraint]
        
        regional_policies['composite_score'] = (
            regional_policies['co2_reduction_mt_per_year'] * 0.3 +
            regional_policies['cost_effectiveness'] * 0.2 +
            regional_policies['feasibility_score'] * 0.2 +
            regional_policies['success_rate_percent'] * 0.15 +
            regional_policies['employment_impact_jobs'] / 1000 * 0.15
        )
        
        recommendations = regional_policies.sort_values('composite_score', ascending=False)
        
        return recommendations[['policy_name', 'policy_type', 'co2_reduction_mt_per_year', 
                               'implementation_cost_millions_usd', 'feasibility_score', 
                               'composite_score']]
    
    def get_emissions_forecast(self, region: str, scenario: str = 'business_as_usual') -> pd.DataFrame:
        """
        Generate an emissions forecast for a region under a given scenario.
        Args:
            region (str): Region to forecast emissions for.
            scenario (str): Scenario key from simulation parameters.
        Returns:
            pd.DataFrame: Emissions forecast by year.
        """
        if not self.simulation_params:
            self.load_simulation_parameters()
        
        baseline_emissions = self.simulation_params['regional_parameters'][region.lower().replace(' ', '_')]['baseline_emissions']
        scenario_params = self.simulation_params['scenario_parameters'][scenario]
        
        current_year = 2024
        forecast_years = range(current_year, current_year + 31)  
        
        emissions_forecast = []
        current_emissions = baseline_emissions
        
        for year in forecast_years:
            emissions_growth = scenario_params['emissions_growth']
            current_emissions *= (1 + emissions_growth)
            
            emissions_forecast.append({
                'year': year,
                'region': region,
                'scenario': scenario,
                'emissions_mt': current_emissions,
                'emissions_growth_rate': emissions_growth
            })
        
        return pd.DataFrame(emissions_forecast)
    
    def export_processed_data(self, output_dir: str = "processed_data"):
        """
        Export all cached processed data to CSV or JSON files in the specified output directory.
        Args:
            output_dir (str): Directory to save processed data files.
        """
        if not self.data_cache:
            self.load_all_data()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, data in self.data_cache.items():
            if isinstance(data, pd.DataFrame):
                output_file = output_path / f"{name}.csv"
                data.to_csv(output_file, index=False)
                logger.info(f"Exported {name} to {output_file}")
            elif isinstance(data, dict):
                output_file = output_path / f"{name}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Exported {name} to {output_file}") 