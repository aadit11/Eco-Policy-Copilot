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
        """
        Generate an executive summary for a given region and policy recommendations.
        Args:
            region (str): The region for which to generate the summary.
            policy_recommendations (List[Dict], optional): List of policy recommendations. Defaults to None.
        Returns:
            Dict[str, Any]: Executive summary dictionary or error message.
        """
        self.initialize()
        
        logger.info(f"Generating executive summary for {region}")
        
        if not policy_recommendations:
            policy_recommendations = self.policy_agent.generate_policy_recommendations(region)
        
        if not policy_recommendations:
            return {'error': 'No policy recommendations available'}
        
        baseline_data = self.data_agent.get_climate_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        
        current_emissions = baseline_data.groupby('year', observed=False)['co2_emissions_mt'].sum().iloc[-1] if not baseline_data.empty else 0
        current_gdp = economic_data.groupby('region', observed=False)['gdp_billions_usd'].last().iloc[0] if not economic_data.empty else 0
        
        best_policy = policy_recommendations[0]
        
        summary = {
            'region': region,
            'current_situation': self._describe_current_situation(region, current_emissions, current_gdp),
            'recommended_policy': self._describe_policy_recommendation(best_policy),
            'expected_impacts': self._describe_expected_impacts(best_policy, current_emissions),
            'implementation_plan': self._describe_implementation_plan(best_policy),
            'risk_assessment': self._describe_risks(best_policy),
            'next_steps': self._generate_next_steps(best_policy),
            'generated_at': datetime.now().isoformat()
        }
        
        self.communication_history.append(summary)
        return summary
    
    def _describe_current_situation(self, region: str, current_emissions: float, current_gdp: float) -> Dict[str, str]:
        """
        Describe the current climate and economic situation for a region.
        Args:
            region (str): The region name.
            current_emissions (float): Current CO2 emissions.
            current_gdp (float): Current GDP.
        Returns:
            Dict[str, str]: Situation description.
        """
        emissions_trend = self.data_agent.get_emissions_trend(region)
        energy_mix = self.data_agent.get_energy_mix_analysis(region)
        economic_indicators = self.data_agent.get_economic_indicators(region)
        
        situation = {
            'emissions_status': f"{region} currently emits {current_emissions:.1f} million tons of CO2 annually. ",
            'emissions_trend': f"Emissions are {emissions_trend.get('trend_direction', 'stable')} with a {emissions_trend.get('emissions_change_percent', 0):.1f}% change over the past decade. ",
            'energy_mix': f"Renewable energy currently represents {energy_mix.get('renewable_share_percent', 0):.1f}% of the energy mix. ",
            'economic_context': f"The region's GDP is ${current_gdp:.1f} billion with a growth rate of {economic_indicators.get('gdp_growth_rate', 0):.1f}%. ",
            'urgency_level': self._assess_urgency_level(emissions_trend, energy_mix)
        }
        
        return situation
    
    def _describe_policy_recommendation(self, policy: Dict) -> Dict[str, str]:
        """
        Describe the recommended policy package.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            Dict[str, str]: Policy description.
        """
        policies = policy['policies']
        
        description = {
            'overview': f"This policy package includes {len(policies)} complementary measures designed to achieve a {policy['emission_reduction_percent']:.1f}% reduction in emissions. ",
            'key_policies': self._list_key_policies(policies),
            'cost_benefit': f"The total implementation cost is ${policy['total_cost_millions_usd']:.1f} million, with net benefits of ${policy['net_benefit_millions_usd']:.1f} million over the policy period. ",
            'cost_effectiveness': f"This represents a cost-effectiveness ratio of {policy['cost_effectiveness']:.2f} tons of CO2 reduced per million dollars invested. ",
            'feasibility': f"The policy package has a feasibility score of {policy['feasibility_score']:.1f}/10, indicating {'high' if policy['feasibility_score'] > 7 else 'moderate' if policy['feasibility_score'] > 4 else 'low'} political and public acceptance potential. "
        }
        
        return description
    
    def _describe_expected_impacts(self, policy: Dict, current_emissions: float) -> Dict[str, str]:
        """
        Describe the expected impacts of the recommended policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
            current_emissions (float): Current CO2 emissions.
        Returns:
            Dict[str, str]: Expected impacts description.
        """
        impacts = {
            'emission_reduction': f"Expected to reduce emissions by {policy['emission_reduction_percent']:.1f}% ({policy['total_cost_millions_usd'] * policy['cost_effectiveness']:.1f} million tons of CO2). ",
            'economic_impact': self._describe_economic_impact(policy),
            'social_benefits': self._describe_social_benefits(policy),
            'environmental_benefits': self._describe_environmental_benefits(policy),
            'timeline': f"Full implementation will take {policy['implementation_timeline']['total_duration_years']} years to achieve maximum impact. "
        }
        
        return impacts
    
    def _describe_implementation_plan(self, policy: Dict) -> Dict[str, Any]:
        """
        Describe the implementation plan for the recommended policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            Dict[str, Any]: Implementation plan details.
        """
        timeline = policy['implementation_timeline']
        
        plan = {
            'total_duration': f"{timeline['total_duration_years']} years",
            'phases': []
        }
        
        for phase in timeline['phases']:
            plan['phases'].append({
                'phase_name': phase['phase'],
                'duration': f"{phase['duration_months']} months",
                'key_activities': ', '.join(phase['activities']),
                'description': f"During the {phase['phase'].lower()} phase, the focus will be on {', '.join(phase['activities']).lower()}. "
            })
        
        return plan
    
    def _describe_risks(self, policy: Dict) -> Dict[str, str]:
        """
        Describe the risks associated with the recommended policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            Dict[str, str]: Risk assessment description.
        """
        risks = policy['risk_assessment']
        
        risk_description = {
            'overall_risk': f"Overall risk assessment indicates {'high' if any(r == 'high' for r in risks.values() if isinstance(r, str)) else 'moderate'} risk levels. ",
            'political_risk': f"Political risk is {risks['political_risk']}, requiring careful stakeholder engagement. ",
            'economic_risk': f"Economic risk is {risks['economic_risk']}, with potential impacts on business competitiveness. ",
            'implementation_risk': f"Implementation risk is {risks['implementation_risk']}, necessitating robust project management. ",
            'mitigation_strategies': f"Key mitigation strategies include: {', '.join(risks['mitigation_strategies'])}. "
        }
        
        return risk_description
    
    def _generate_next_steps(self, policy: Dict) -> List[str]:
        """
        Generate a list of next steps for policy implementation.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            List[str]: List of next steps.
        """
        steps = [
            "Conduct detailed stakeholder consultation and public engagement",
            "Develop comprehensive implementation roadmap with milestones",
            "Establish monitoring and evaluation framework",
            "Secure necessary funding and budget allocations",
            "Begin legislative and regulatory development process",
            "Set up cross-departmental coordination mechanisms"
        ]
        
        return steps
    
    def _assess_urgency_level(self, emissions_trend: Dict, energy_mix: Dict) -> str:
        """
        Assess the urgency level for climate action based on emissions trend and energy mix.
        Args:
            emissions_trend (Dict): Emissions trend data.
            energy_mix (Dict): Energy mix data.
        Returns:
            str: Urgency level description.
        """
        if emissions_trend.get('trend_direction') == 'increasing' and energy_mix.get('renewable_share_percent', 0) < 20:
            return "High urgency - immediate action required to address rising emissions and low renewable energy adoption."
        elif emissions_trend.get('trend_direction') == 'increasing':
            return "Moderate urgency - emissions are increasing but some renewable energy progress is being made."
        elif energy_mix.get('renewable_share_percent', 0) < 30:
            return "Moderate urgency - emissions are stable but renewable energy adoption needs acceleration."
        else:
            return "Lower urgency - good progress on emissions and renewable energy, focus on optimization."
    
    def _list_key_policies(self, policies: List[Dict]) -> str:
        """
        List key policies in a policy package.
        Args:
            policies (List[Dict]): List of policy dictionaries.
        Returns:
            str: Comma-separated key policy descriptions.
        """
        policy_descriptions = []
        
        for policy in policies:
            if policy['type'] == 'carbon_tax':
                desc = f"Carbon tax at ${policy.get('tax_rate_usd_per_ton', 0)}/ton"
            elif policy['type'] == 'renewable_subsidy':
                desc = f"Renewable energy subsidy at {policy.get('subsidy_rate_percent', 0)}%"
            elif policy['type'] == 'energy_efficiency':
                desc = f"Energy efficiency improvements of {policy.get('efficiency_improvement_percent', 0)}%"
            elif policy['type'] == 'ev_subsidy':
                desc = f"EV adoption target of {policy.get('ev_penetration_target_percent', 0)}%"
            else:
                desc = policy.get('description', policy['name'])
            
            policy_descriptions.append(desc)
        
        return ', '.join(policy_descriptions)
    
    def _describe_economic_impact(self, policy: Dict) -> str:
        """
        Describe the economic impact of a policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            str: Economic impact description.
        """
        net_benefit = policy['net_benefit_millions_usd']
        total_cost = policy['total_cost_millions_usd']
        
        if net_benefit > 0:
            return f"Positive economic impact with net benefits of ${net_benefit:.1f} million, representing a {net_benefit/total_cost*100:.1f}% return on investment. "
        else:
            return f"Economic cost of ${abs(net_benefit):.1f} million, representing {abs(net_benefit)/total_cost*100:.1f}% of total investment cost. "
    
    def _describe_social_benefits(self, policy: Dict) -> str:
        """
        Describe the social benefits of a policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            str: Social benefits description.
        """
        benefits = [
            "Improved public health through reduced air pollution",
            "Enhanced energy security and independence",
            "Job creation in clean energy sectors",
            "Reduced energy costs for consumers over time",
            "Improved quality of life and community resilience"
        ]
        
        return f"Social benefits include: {', '.join(benefits)}. "
    
    def _describe_environmental_benefits(self, policy: Dict) -> str:
        """
        Describe the environmental benefits of a policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            str: Environmental benefits description.
        """
        emission_reduction = policy['emission_reduction_percent']
        
        benefits = [
            f"Significant reduction in greenhouse gas emissions ({emission_reduction:.1f}%)",
            "Improved air quality and reduced local pollution",
            "Enhanced ecosystem health and biodiversity",
            "Progress toward climate action goals",
            "Reduced environmental degradation and resource depletion"
        ]
        
        return f"Environmental benefits include: {', '.join(benefits)}. "
    
    def generate_policy_brief(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate a detailed policy brief for a region and policy recommendations.
        Args:
            region (str): The region for which to generate the brief.
            policy_recommendations (List[Dict], optional): List of policy recommendations. Defaults to None.
        Returns:
            Dict[str, Any]: Policy brief dictionary or error message.
        """
        self.initialize()
        
        if not policy_recommendations:
            policy_recommendations = self.policy_agent.generate_policy_recommendations(region)
        
        if not policy_recommendations:
            return {'error': 'No policy recommendations available'}
        
        brief = {
            'title': f"Climate Policy Recommendations for {region}",
            'executive_summary': self.generate_executive_summary(region, policy_recommendations),
            'detailed_analysis': self._generate_detailed_analysis(region, policy_recommendations),
            'comparison_table': self._generate_comparison_table(policy_recommendations),
            'implementation_roadmap': self._generate_implementation_roadmap(policy_recommendations[0]),
            'appendices': self._generate_appendices(region, policy_recommendations)
        }
        
        return brief
    
    def _generate_detailed_analysis(self, region: str, policy_recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Generate a detailed analysis section for the policy brief.
        Args:
            region (str): The region name.
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, Any]: Detailed analysis dictionary.
        """
        analysis = {
            'regional_context': self._analyze_regional_context(region),
            'policy_effectiveness': self._analyze_policy_effectiveness(policy_recommendations),
            'economic_analysis': self._analyze_economic_impacts(policy_recommendations),
            'stakeholder_analysis': self._analyze_stakeholder_impacts(policy_recommendations),
            'risk_mitigation': self._analyze_risk_mitigation(policy_recommendations)
        }
        
        return analysis
    
    def _analyze_regional_context(self, region: str) -> Dict[str, str]:
        """
        Analyze the regional context for climate policy.
        Args:
            region (str): The region name.
        Returns:
            Dict[str, str]: Regional context analysis.
        """
        climate_data = self.data_agent.get_climate_data(region)
        economic_data = self.data_agent.get_economic_data(region)
        energy_data = self.data_agent.get_energy_data(region)
        
        context = {
            'emissions_profile': f"{region} has a diverse emissions profile with {len(climate_data['sector'].unique())} major sectors contributing to total emissions. ",
            'economic_strength': f"The region's economic strength provides capacity for climate investment, with GDP per capita of ${economic_data['gdp_per_capita'].iloc[-1]:.0f}. ",
            'energy_transition': f"Current renewable energy share of {energy_data['renewable_share'].iloc[-1]:.1f}% indicates {'strong' if energy_data['renewable_share'].iloc[-1] > 30 else 'moderate' if energy_data['renewable_share'].iloc[-1] > 15 else 'limited'} progress in energy transition. ",
            'policy_readiness': self._assess_policy_readiness(region)
        }
        
        return context
    
    def _analyze_policy_effectiveness(self, policy_recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of policy recommendations.
        Args:
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, Any]: Policy effectiveness analysis.
        """
        effectiveness = {
            'emission_reduction_potential': f"Policy packages can achieve {policy_recommendations[0]['emission_reduction_percent']:.1f}% to {policy_recommendations[-1]['emission_reduction_percent']:.1f}% emission reductions. ",
            'cost_effectiveness_range': f"Cost-effectiveness ranges from {policy_recommendations[0]['cost_effectiveness']:.2f} to {policy_recommendations[-1]['cost_effectiveness']:.2f} tons CO2 per million dollars. ",
            'feasibility_considerations': f"Feasibility scores range from {policy_recommendations[-1]['feasibility_score']:.1f} to {policy_recommendations[0]['feasibility_score']:.1f}/10. ",
            'implementation_timeline': f"Implementation timelines range from {policy_recommendations[0]['implementation_timeline']['total_duration_years']} to {policy_recommendations[-1]['implementation_timeline']['total_duration_years']} years. "
        }
        
        return effectiveness
    
    def _analyze_economic_impacts(self, policy_recommendations: List[Dict]) -> Dict[str, str]:
        """
        Analyze the economic impacts of policy recommendations.
        Args:
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, str]: Economic impact analysis.
        """
        total_costs = [p['total_cost_millions_usd'] for p in policy_recommendations]
        net_benefits = [p['net_benefit_millions_usd'] for p in policy_recommendations]
        
        analysis = {
            'investment_scale': f"Total investment required ranges from ${min(total_costs):.1f} to ${max(total_costs):.1f} million. ",
            'economic_return': f"Net economic benefits range from ${min(net_benefits):.1f} to ${max(net_benefits):.1f} million. ",
            'cost_distribution': f"Investment costs are distributed across {len(policy_recommendations[0]['policies'])} policy areas with varying implementation timelines. ",
            'economic_viability': f"{'All' if all(b > 0 for b in net_benefits) else 'Most' if sum(1 for b in net_benefits if b > 0) > len(net_benefits)/2 else 'Some'} policy packages show positive economic returns. "
        }
        
        return analysis
    
    def _analyze_stakeholder_impacts(self, policy_recommendations: List[Dict]) -> Dict[str, str]:
        """
        Analyze the stakeholder impacts of policy recommendations.
        Args:
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, str]: Stakeholder impact analysis.
        """
        analysis = {
            'business_impact': "Businesses will face initial compliance costs but benefit from long-term energy savings and market opportunities in clean technologies. ",
            'consumer_impact': "Consumers may experience short-term cost increases but will benefit from improved air quality and long-term energy cost reductions. ",
            'government_role': "Government will need to provide leadership, funding, and regulatory frameworks while ensuring equitable distribution of benefits and costs. ",
            'community_benefits': "Communities will benefit from improved public health, job creation, and enhanced resilience to climate impacts. "
        }
        
        return analysis
    
    def _analyze_risk_mitigation(self, policy_recommendations: List[Dict]) -> Dict[str, str]:
        """
        Analyze risk mitigation strategies for policy recommendations.
        Args:
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, str]: Risk mitigation analysis.
        """
        analysis = {
            'political_risk_mitigation': "Engage stakeholders early, provide clear communication, and implement policies gradually to build public support. ",
            'economic_risk_mitigation': "Use phased implementation, provide transition support, and establish monitoring systems to track economic impacts. ",
            'implementation_risk_mitigation': "Develop robust project management frameworks, provide technical assistance, and establish clear accountability mechanisms. ",
            'monitoring_framework': "Establish comprehensive monitoring and evaluation systems to track progress and enable adaptive management. "
        }
        
        return analysis
    
    def _assess_policy_readiness(self, region: str) -> str:
        """
        Assess the policy readiness of a region.
        Args:
            region (str): The region name.
        Returns:
            str: Policy readiness description.
        """
        policy_data = self.data_agent.get_policy_data(region)
        
        if policy_data.empty:
            return "Limited policy experience in this region, requiring capacity building and stakeholder engagement."
        
        avg_feasibility = policy_data['feasibility_score'].mean()
        if avg_feasibility > 7:
            return "High policy readiness with strong stakeholder support and implementation capacity."
        elif avg_feasibility > 4:
            return "Moderate policy readiness with some stakeholder support and implementation experience."
        else:
            return "Low policy readiness requiring significant capacity building and stakeholder engagement."
    
    def _generate_comparison_table(self, policy_recommendations: List[Dict]) -> List[Dict]:
        """
        Generate a comparison table for up to five policy recommendations.
        Args:
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            List[Dict]: Comparison table rows.
        """
        comparison = []
        
        for i, policy in enumerate(policy_recommendations[:5]):
            comparison.append({
                'option': f"Option {i+1}",
                'emission_reduction': f"{policy['emission_reduction_percent']:.1f}%",
                'total_cost': f"${policy['total_cost_millions_usd']:.1f}M",
                'net_benefit': f"${policy['net_benefit_millions_usd']:.1f}M",
                'cost_effectiveness': f"{policy['cost_effectiveness']:.2f}",
                'feasibility': f"{policy['feasibility_score']:.1f}/10",
                'timeline': f"{policy['implementation_timeline']['total_duration_years']} years"
            })
        
        return comparison
    
    def _generate_implementation_roadmap(self, policy: Dict) -> Dict[str, Any]:
        """
        Generate an implementation roadmap for a policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            Dict[str, Any]: Implementation roadmap details.
        """
        roadmap = {
            'overview': f"Implementation roadmap for achieving {policy['emission_reduction_percent']:.1f}% emission reduction over {policy['implementation_timeline']['total_duration_years']} years. ",
            'phases': policy['implementation_timeline']['phases'],
            'key_milestones': self._generate_key_milestones(policy),
            'success_metrics': self._generate_success_metrics(policy)
        }
        
        return roadmap
    
    def _generate_key_milestones(self, policy: Dict) -> List[Dict]:
        """
        Generate key milestones for a policy's implementation timeline.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            List[Dict]: List of milestone dictionaries.
        """
        timeline = policy['implementation_timeline']
        milestones = []
        
        cumulative_months = 0
        for phase in timeline['phases']:
            cumulative_months += phase['duration_months']
            milestones.append({
                'milestone': f"Complete {phase['phase']} phase",
                'timeline': f"Month {cumulative_months}",
                'deliverables': f"Completed {', '.join(phase['activities']).lower()}"
            })
        
        return milestones
    
    def _generate_success_metrics(self, policy: Dict) -> List[Dict]:
        """
        Generate success metrics for a policy.
        Args:
            policy (Dict): Policy recommendation dictionary.
        Returns:
            List[Dict]: List of success metric dictionaries.
        """
        metrics = [
            {
                'metric': 'Emission Reduction',
                'target': f"{policy['emission_reduction_percent']:.1f}%",
                'measurement': 'Annual CO2 emissions tracking'
            },
            {
                'metric': 'Economic Return',
                'target': f"${policy['net_benefit_millions_usd']:.1f}M net benefit",
                'measurement': 'Cost-benefit analysis'
            },
            {
                'metric': 'Implementation Progress',
                'target': '100% policy implementation',
                'measurement': 'Project milestone tracking'
            },
            {
                'metric': 'Stakeholder Satisfaction',
                'target': 'High satisfaction scores',
                'measurement': 'Regular stakeholder surveys'
            }
        ]
        
        return metrics
    
    def _generate_appendices(self, region: str, policy_recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Generate appendices for the policy brief.
        Args:
            region (str): The region name.
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, Any]: Appendices dictionary.
        """
        appendices = {
            'technical_details': self._generate_technical_details(region, policy_recommendations),
            'data_sources': self._generate_data_sources(region),
            'methodology': self._generate_methodology(),
            'glossary': self._generate_glossary()
        }
        
        return appendices
    
    def _generate_technical_details(self, region: str, policy_recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Generate technical details for the appendices.
        Args:
            region (str): The region name.
            policy_recommendations (List[Dict]): List of policy recommendations.
        Returns:
            Dict[str, Any]: Technical details dictionary.
        """
        details = {
            'simulation_parameters': self.simulation_agent.get_simulation_metadata(),
            'data_summary': self.data_agent.get_data_summary(),
            'policy_metadata': self.policy_agent.get_policy_metadata(),
            'regional_analysis': self.data_agent.get_regional_summary(region)
        }
        
        return details
    
    def _generate_data_sources(self, region: str) -> List[str]:
        """
        Generate a list of data sources for the appendices.
        Args:
            region (str): The region name.
        Returns:
            List[str]: List of data source descriptions.
        """
        sources = [
            "Regional climate data from national environmental agencies",
            "Economic indicators from statistical bureaus",
            "Energy consumption data from energy authorities",
            "Policy effectiveness data from international databases",
            "Social impact assessments from research institutions"
        ]
        
        return sources
    
    def _generate_methodology(self) -> Dict[str, str]:
        """
        Generate methodology details for the appendices.
        Returns:
            Dict[str, str]: Methodology description.
        """
        methodology = {
            'simulation_approach': "Integrated climate-economic modeling using baseline projections and policy impact analysis. ",
            'policy_evaluation': "Multi-criteria analysis considering effectiveness, cost, feasibility, and implementation timeline. ",
            'stakeholder_analysis': "Comprehensive assessment of political, economic, and social factors affecting policy success. ",
            'risk_assessment': "Systematic evaluation of implementation risks and mitigation strategies. "
        }
        
        return methodology
    
    def _generate_glossary(self) -> Dict[str, str]:
        """
        Generate a glossary for the appendices.
        Returns:
            Dict[str, str]: Glossary dictionary.
        """
        glossary = {
            'CO2 emissions': "Carbon dioxide emissions, a primary greenhouse gas contributing to climate change",
            'Cost-effectiveness': "Measure of policy efficiency in terms of emission reduction per dollar invested",
            'Feasibility score': "Assessment of political and public acceptance potential for policy implementation",
            'Net benefit': "Economic benefits minus costs over the policy implementation period",
            'Renewable energy': "Energy from sources that are naturally replenished, such as solar, wind, and hydro"
        }
        
        return glossary
    
    def export_communications(self, output_dir: str = "communications_output"):
        """
        Export all generated communications to the specified output directory as JSON files.
        Args:
            output_dir (str, optional): Output directory. Defaults to "communications_output".
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, communication in enumerate(self.communication_history):
            filename = f"executive_summary_{communication['region']}_{timestamp}.json"
            with open(output_path / filename, 'w') as f:
                json.dump(communication, f, indent=2)
        
        logger.info(f"Communications exported to {output_path}")
    
    def get_communications_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about all generated communications.
        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        metadata = {
            'total_communications': len(self.communication_history),
            'regions_covered': list(set([c['region'] for c in self.communication_history])),
            'last_communication': max([c['generated_at'] for c in self.communication_history], default=None),
            'communication_types': ['executive_summary', 'policy_brief', 'detailed_analysis']
        }
        
        return metadata
    
    def generate_llm_executive_summary(self, region: str, policy_recommendations: List[Dict] = None) -> Dict[str, str]:
        """
        Generate an executive summary using an LLM for a given region and policy recommendations.
        Args:
            region (str): The region for which to generate the summary.
            policy_recommendations (List[Dict], optional): List of policy recommendations. Defaults to None.
        Returns:
            Dict[str, str]: LLM-generated executive summary or error message.
        """
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
        """
        Generate a policy brief using an LLM for a given region and policy recommendations.
        Args:
            region (str): The region for which to generate the policy brief.
            policy_recommendations (List[Dict], optional): List of policy recommendations. Defaults to None.
        Returns:
            Dict[str, str]: LLM-generated policy brief or error message.
        """
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
