SETTINGS = {
    "data_agent": {
        "data_dir": "data"
    },
    "simulation_agent": {
        "default_years": 20
    },
    "policy_generator_agent": {
        "default_target_emission_reduction": 30,
        "default_budget_constraint": None,
        "default_policy_count": 5
    },
    "communications_agent": {
        "output_dir": "communications_output"
    },
    "workflow": {
        "max_iterations": 5,
        "convergence_threshold": 0.5
    },
    "llm": {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "endpoint": "http://localhost:11434/api/generate",
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 60
    }
}
