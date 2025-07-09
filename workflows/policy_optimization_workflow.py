from langgraph.graph import StateGraph, END, START
from agents.data_agent import DataAgent
from agents.simulation_agent import SimulationAgent
from agents.policy_agent import PolicyGeneratorAgent
from agents.communications_agent import CommunicationsAgent
from typing_extensions import TypedDict

class PolicyState(TypedDict, total=False):
    region: str
    target_emission_reduction: float
    budget_constraint: float
    iteration: int
    max_iterations: int
    converged: bool
    previous_emission_reduction: float
    best_policy: dict
    best_result: dict
    policy_recommendations: list
    simulation_result: dict
    history: list
    final_output: dict
    llm_insights: dict


def data_node(state: PolicyState, data_agent: DataAgent):
    data_agent.initialize()
    llm_insights = data_agent.get_llm_data_insights(state["region"])
    state["llm_insights"] = {"data_insights": llm_insights}
    return state

def policy_node(state: PolicyState, policy_agent: PolicyGeneratorAgent):
    llm_policy_recs = policy_agent.get_llm_policy_recommendations(
        state["region"], 
        state["target_emission_reduction"]
    )
    
    policy_recommendations = policy_agent.generate_policy_recommendations(
        state["region"],
        target_emission_reduction=state["target_emission_reduction"],
        budget_constraint=state["budget_constraint"],
        policy_count=3
    )
    
    state["policy_recommendations"] = policy_recommendations
    if policy_recommendations:
        state["best_policy"] = policy_recommendations[0]
        
        llm_optimization = policy_agent.get_llm_policy_optimization(
            state["region"], 
            policy_recommendations
        )
        
        if state.get("llm_insights") is None:
            state["llm_insights"] = {}
        state["llm_insights"].update({
            'policy_recommendations': llm_policy_recs,
            'policy_optimization': llm_optimization
        })
    
    return state

def simulation_node(state: PolicyState, simulation_agent: SimulationAgent):
    if state.get("best_policy"):
        simulation_result = simulation_agent.run_policy_simulation(state["region"], state["best_policy"]["policies"])
        state["simulation_result"] = simulation_result
        state["best_result"] = simulation_result
        
        llm_analysis = simulation_agent.get_llm_simulation_analysis(state["region"], simulation_result)
        
        if state.get("llm_insights") is None:
            state["llm_insights"] = {}
        state["llm_insights"]["simulation_analysis"] = llm_analysis
    
    return state

def convergence_node(state: PolicyState):
    if not state.get("best_policy"):
        state["converged"] = True
        return state
    
    emission_reduction = state["best_policy"]["emission_reduction_percent"]
    if abs(emission_reduction - state.get("previous_emission_reduction", 0.0)) < 0.5:
        state["converged"] = True
    
    state["previous_emission_reduction"] = emission_reduction
    state["iteration"] = state.get("iteration", 0) + 1
    if state["iteration"] >= state.get("max_iterations", 5):
        state["converged"] = True
    
    return state

def communications_node(state: PolicyState, communications_agent: CommunicationsAgent):
    if state.get("best_policy"):
        state["final_output"] = communications_agent.generate_policy_brief(state["region"], [state["best_policy"]])
        
        llm_summary = communications_agent.generate_llm_executive_summary(
            state["region"], 
            [state["best_policy"]]
        )
        
        llm_brief = communications_agent.generate_llm_policy_brief(
            state["region"], 
            [state["best_policy"]]
        )
        
        if state.get("llm_insights") is None:
            state["llm_insights"] = {}
        state["llm_insights"].update({
            'executive_summary': llm_summary,
            'policy_brief': llm_brief
        })
        
        if state["final_output"]:
            state["final_output"]["llm_enhanced_outputs"] = state["llm_insights"]
    
    return state

def build_policy_optimization_graph(region, target_emission_reduction=30, budget_constraint=None, max_iterations=5):
    data_agent = DataAgent()
    simulation_agent = SimulationAgent(data_agent)
    policy_agent = PolicyGeneratorAgent(data_agent, simulation_agent)
    communications_agent = CommunicationsAgent(data_agent, simulation_agent, policy_agent)

    graph = StateGraph(PolicyState)
    graph.add_node("data", lambda state: data_node(state, data_agent))
    graph.add_node("policy", lambda state: policy_node(state, policy_agent))
    graph.add_node("simulation", lambda state: simulation_node(state, simulation_agent))
    graph.add_node("convergence", convergence_node)
    graph.add_node("communications", lambda state: communications_node(state, communications_agent))

    graph.add_edge(START, "data")
    graph.add_edge("data", "policy")
    graph.add_edge("policy", "simulation")
    graph.add_edge("simulation", "convergence")
    graph.add_conditional_edges(
        "convergence",
        lambda state: "communications" if state.get("converged") else "policy"
    )
    graph.add_edge("communications", END)

    workflow = graph.compile()
    initial_state = PolicyState(
        region=region,
        target_emission_reduction=target_emission_reduction,
        budget_constraint=budget_constraint,
        max_iterations=max_iterations,
        iteration=0,
        converged=False,
        previous_emission_reduction=0.0,
        best_policy=None,
        best_result=None,
        policy_recommendations=None,
        simulation_result=None,
        history=[],
        final_output=None,
        llm_insights=None
    )
    return workflow, initial_state
