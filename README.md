# 🌱 Eco-Policy-Copilot

An intelligent AI-powered system for generating, optimizing, and simulating environmental policy recommendations using multi-agent workflows and advanced data analysis.

## 🎯 Overview

Eco-Policy-Copilot is a sophisticated policy optimization system that leverages artificial intelligence to generate data-driven environmental policy recommendations. The system uses a multi-agent architecture to analyze climate data, simulate policy impacts, and generate comprehensive policy briefs with executive summaries.

### Key Capabilities

- **Data-Driven Analysis**: Processes climate, economic, and energy data to understand regional contexts
- **Policy Generation**: Creates optimized policy recommendations based on emission reduction targets
- **Impact Simulation**: Simulates policy outcomes using advanced modeling techniques
- **Intelligent Optimization**: Uses iterative optimization to find the best policy combinations
- **Professional Output**: Generates policy briefs and executive summaries for stakeholders

## ✨ Features

### 🔍 Data Analysis
- Multi-dimensional data processing (climate, economic, energy, social impact)
- Regional trend analysis and forecasting
- Comparative analysis across regions
- Real-time data insights using LLM integration

### 🎯 Policy Optimization
- Target-based emission reduction planning
- Budget-constrained policy selection
- Multi-criteria optimization (cost-effectiveness, feasibility, impact)
- Iterative refinement with convergence tracking

### 📊 Simulation & Modeling
- Policy impact simulation over 20-year horizons
- Economic cost-benefit analysis
- Social and environmental impact assessment
- Risk assessment and mitigation strategies

### 📝 Communication & Reporting
- Automated policy brief generation
- Executive summary creation
- Stakeholder-ready documentation
- Multiple output formats (JSON, TXT)

## 🏗️ Architecture

The system uses a **multi-agent workflow architecture** with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Agent    │    │  Policy Agent   │    │ Simulation Agent│
│                 │    │                 │    │                 │
│ • Data Loading  │───▶│ • Policy Gen    │───▶│ • Impact Sim    │
│ • Analysis      │    │ • Optimization  │    │ • Modeling      │
│ • Insights      │    │ • Refinement    │    │ • Forecasting   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │Communications   │
                    │Agent            │
                    │                 │
                    │ • Policy Briefs │
                    │ • Exec Summary  │
                    │ • Stakeholder   │
                    │   Reports       │
                    └─────────────────┘
```

### Core Components

1. **Data Agent** (`agents/data_agent.py`)
   - Manages data loading and caching
   - Provides regional data analysis
   - Generates LLM-powered insights

2. **Policy Agent** (`agents/policy_agent.py`)
   - Generates policy recommendations
   - Optimizes policy combinations
   - Handles budget constraints

3. **Simulation Agent** (`agents/simulation_agent.py`)
   - Simulates policy impacts
   - Models economic outcomes
   - Assesses risks and benefits

4. **Communications Agent** (`agents/communications_agent.py`)
   - Creates policy briefs
   - Generates executive summaries
   - Formats stakeholder reports

5. **Workflow Engine** (`workflows/policy_optimization_workflow.py`)
   - Orchestrates agent interactions
   - Manages optimization iterations
   - Handles convergence logic

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Ollama (for local LLM inference)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eco-policy-copilot.git
   cd eco-policy-copilot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (for LLM functionality)
   ```bash
   # Follow instructions at https://ollama.ai
   ollama pull llama3.1:8b
   ```

5. **Verify installation**
   ```bash
   python main.py --help
   ```

## 📊 Data Sources

The system uses the following data files in the `data/` directory:

- **`climate_data.csv`**: Regional CO2 emissions and climate metrics
- **`energy_data.csv`**: Energy consumption by source and region
- **`economic_data.csv`**: GDP, population, and economic indicators
- **`policy_database.csv`**: Historical policy effectiveness data
- **`social_impact_data.csv`**: Social and community impact metrics
- **`technology_cost_data.csv`**: Technology cost and adoption data
- **`simulation_parameters.json`**: Simulation configuration parameters


