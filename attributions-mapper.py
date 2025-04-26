attributions.append({
                        "collapse_id": collapse_point.index,
                        "collapse_type": collapse_type,
                        "source_index": source_index,
                        "source_token": prompt_tokens[source_index],
                        "target_index": collapse_index,
                        "path": path,
                        "intensity": collapse_point.intensity,
                        "confidence": collapse_point.confidence
                    })
        
        return collapse_attributions
    
    def _initialize_flow_graph(self) -> Dict:
        """Initialize an empty attribution flow graph"""
        return {
            "nodes": [],
            "edges": [],
            "groups": [],
            "metadata": {
                "node_count": 0,
                "edge_count": 0,
                "group_count": 0
            }
        }
    
    def _build_flow_graph(
        self,
        direct_paths: List[AttributionPath],
        indirect_paths: List[AttributionPath],
        collapse_attributions: List[Dict]
    ) -> Dict:
        """
        Build a graph representation of attribution flows
        
        This creates a networkx graph and then converts it to a visualization-friendly
        format with nodes, edges, and groupings.
        """
        # Create a networkx directed graph
        G = nx.DiGraph()
        
        # Track node indices
        node_indices = {}
        next_node_id = 0
        
        # Function to add a node if it doesn't exist
        def add_node(label, node_type, index=None, metadata=None):
            nonlocal next_node_id
            
            if (node_type, label) not in node_indices:
                node_id = next_node_id
                next_node_id += 1
                node_indices[(node_type, label)] = node_id
                
                G.add_node(
                    node_id,
                    id=node_id,
                    label=label,
                    type=node_type,
                    index=index,
                    metadata=metadata or {}
                )
            
            return node_indices[(node_type, label)]
        
        # Add nodes and edges from direct paths
        for path in direct_paths:
            source_id = add_node(
                f"Source {path.source_index}",
                "source",
                path.source_index
            )
            target_id = add_node(
                f"Target {path.target_index}",
                "target",
                path.target_index
            )
            
            G.add_edge(
                source_id,
                target_id,
                weight=path.strength,
                type=path.path_type,
                metadata={
                    "strength": path.strength,
                    "path_type": path.path_type
                }
            )
        
        # Add nodes and edges from indirect paths
        for path in indirect_paths:
            source_id = add_node(
                f"Source {path.source_index}",
                "source",
                path.source_index
            )
            target_id = add_node(
                f"Target {path.target_index}",
                "target",
                path.target_index
            )
            
            # Add intermediate nodes if any
            if path.intermediates:
                prev_id = source_id
                
                for intermediate in path.intermediates:
                    intermediate_id = add_node(
                        f"Intermediate {intermediate['index']}",
                        "intermediate",
                        intermediate['index'],
                        intermediate
                    )
                    
                    # Add edge from previous to intermediate
                    G.add_edge(
                        prev_id,
                        intermediate_id,
                        weight=path.strength,
                        type=f"{path.path_type}_segment",
                        metadata={
                            "strength": path.strength,
                            "path_type": path.path_type
                        }
                    )
                    
                    prev_id = intermediate_id
                
                # Add edge from last intermediate to target
                G.add_edge(
                    prev_id,
                    target_id,
                    weight=path.strength,
                    type=f"{path.path_type}_segment",
                    metadata={
                        "strength": path.strength,
                        "path_type": path.path_type
                    }
                )
            else:
                # Direct edge from source to target
                G.add_edge(
                    source_id,
                    target_id,
                    weight=path.strength,
                    type=path.path_type,
                    metadata={
                        "strength": path.strength,
                        "path_type": path.path_type
                    }
                )
        
        # Add nodes and edges from collapse attributions
        for attribution in collapse_attributions:
            source_id = add_node(
                f"Source {attribution['source_index']}",
                "source",
                attribution['source_index']
            )
            
            # Add collapse node
            collapse_id = add_node(
                f"Collapse {attribution['collapse_id']}",
                "collapse",
                attribution['collapse_id'],
                {
                    "collapse_type": attribution['collapse_type'],
                    "intensity": attribution['intensity'],
                    "confidence": attribution['confidence']
                }
            )
            
            # Add edge from source to collapse
            G.add_edge(
                source_id,
                collapse_id,
                weight=attribution['path'].strength,
                type=attribution['path'].path_type,
                metadata={
                    "strength": attribution['path'].strength,
                    "path_type": attribution['path'].path_type
                }
            )
        
        # Create node groups
        groups = []
        
        # Group by node type
        node_types = set(data['type'] for _, data in G.nodes(data=True))
        for i, node_type in enumerate(node_types):
            group = {
                "id": f"group_{i}",
                "label": node_type.capitalize(),
                "type": "node_type",
                "nodes": [
                    node_id for node_id, data in G.nodes(data=True)
                    if data['type'] == node_type
                ]
            }
            groups.append(group)
        
        # Convert to visualization-friendly format
        flow_graph = {
            "nodes": [
                {
                    "id": node_id,
                    "label": data['label'],
                    "type": data['type'],
                    "index": data['index'],
                    "metadata": data['metadata']
                }
                for node_id, data in G.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": data['weight'],
                    "type": data['type'],
                    "metadata": data['metadata']
                }
                for source, target, data in G.edges(data=True)
            ],
            "groups": groups,
            "metadata": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges(),
                "group_count": len(groups)
            }
        }
        
        return flow_graph
    
    def _generate_attribution_summary(self, attribution_map: Dict) -> Dict:
        """Generate a summary of attribution patterns"""
        summary = {
            "prompt_influence": {},
            "result_influence": {},
            "primary_attributions": [],
            "weak_attributions": [],
            "collapse_sources": {},
            "uncertainty_level": 0.0
        }
        
        # Calculate prompt token influence
        prompt_tokens = attribution_map["prompt_tokens"]
        prompt_influence = {i: 0.0 for i in range(len(prompt_tokens))}
        
        for path in attribution_map["direct_paths"] + attribution_map["indirect_paths"]:
            prompt_influence[path.source_index] += path.strength
        
        for attribution in attribution_map.get("collapse_attributions", []):
            source_index = attribution["source_index"]
            prompt_influence[source_index] += attribution["path"].strength
        
        # Normalize prompt influence
        if prompt_influence:
            max_influence = max(prompt_influence.values())
            if max_influence > 0:
                for i in prompt_influence:
                    prompt_influence[i] /= max_influence
        
        # Sort by influence and add to summary
        sorted_prompt = sorted(
            prompt_influence.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for idx, influence in sorted_prompt:
            if influence > 0.1:  # Only include meaningful influence
                summary["prompt_influence"][idx] = {
                    "token": prompt_tokens[idx],
                    "influence": influence
                }
        
        # Calculate result token influence
        result_tokens = attribution_map["result_tokens"]
        result_influence = {i: 0.0 for i in range(len(result_tokens))}
        
        for path in attribution_map["direct_paths"] + attribution_map["indirect_paths"]:
            result_influence[path.target_index] += path.strength
        
        # Normalize result influence
        if result_influence:
            max_influence = max(result_influence.values())
            if max_influence > 0:
                for i in result_influence:
                    result_influence[i] /= max_influence
        
        # Sort by influence and add to summary
        sorted_result = sorted(
            result_influence.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for idx, influence in sorted_result:
            if influence > 0.1:  # Only include meaningful influence
                summary["result_influence"][idx] = {
                    "token": result_tokens[idx],
                    "influence": influence
                }
        
        # Identify primary attributions (strongest direct paths)
        primary_paths = sorted(
            attribution_map["direct_paths"],
            key=lambda path: path.strength,
            reverse=True
        )[:5]  # Top 5
        
        for path in primary_paths:
            summary["primary_attributions"].append({
                "source_index": path.source_index,
                "source_token": prompt_tokens[path.source_index],
                "target_index": path.target_index,
                "target_token": result_tokens[path.target_index],
                "strength": path.strength,
                "path_type": path.path_type
            })
        
        # Identify weak attributions (connections that seem uncertain)
        weak_paths = [
            path for path in attribution_map["direct_paths"] + attribution_map["indirect_paths"]
            if 0.3 < path.strength < 0.5
        ]
        
        for path in weak_paths[:5]:  # Top 5
            summary["weak_attributions"].append({
                "source_index": path.source_index,
                "source_token": prompt_tokens[path.source_index],
                "target_index": path.target_index,
                "target_token": result_tokens[path.target_index],
                "strength": path.strength,
                "path_type": path.path_type
            })
        
        # Calculate collapse sources (prompt tokens most responsible for collapses)
        collapse_sources = {}
        
        for attribution in attribution_map.get("collapse_attributions", []):
            collapse_type = attribution["collapse_type"]
            source_index = attribution["source_index"]
            strength = attribution["path"].strength
            
            if collapse_type not in collapse_sources:
                collapse_sources[collapse_type] = {}
            
            if source_index not in collapse_sources[collapse_type]:
                collapse_sources[collapse_type][source_index] = 0
            
            collapse_sources[collapse_type][source_index] += strength
        
        # Add top sources for each collapse type
        for collapse_type, sources in collapse_sources.items():
            sorted_sources = sorted(
                sources.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            summary["collapse_sources"][collapse_type] = [
                {
                    "source_index": idx,
                    "source_token": prompt_tokens[idx],
                    "influence": strength
                }
                for idx, strength in sorted_sources[:3]  # Top 3
            ]
        
        # Calculate overall uncertainty level
        # Based on proportion of weak vs. strong attributions and presence of collapses
        path_strengths = [path.strength for path in attribution_map["direct_paths"] + attribution_map["indirect_paths"]]
        if path_strengths:
            # Calculate strength variability
            strength_std = np.std(path_strengths) if len(path_strengths) > 1 else 0
            mean_strength = np.mean(path_strengths)
            
            # Higher std and lower mean = higher uncertainty
            uncertainty_base = 0.5 * strength_std + (1 - mean_strength) * 0.5
            
            # Adjust for collapse points
            collapse_factor = len(attribution_map.get("collapse_attributions", [])) * 0.1
            
            summary["uncertainty_level"] = min(uncertainty_base + collapse_factor, 1.0)
        else:
            summary["uncertainty_level"] = 0.5  # Default
        
        return summary


# ric_core/shells/shell_registry.py

"""
Registry and management system for recursive diagnostic shells.

This module provides the registry for all available diagnostic shells,
allowing users to discover, select, and manage the shells used for
inducing and tracing specific failure modes in transformer models.
"""

from typing import Dict, List, Optional, Any, Type
import logging
import importlib
import pkgutil
import inspect
from .shell_base import RecursiveShell

logger = logging.getLogger("ric.shells")

# Dictionary to store registered shells
SHELL_REGISTRY = {}

def register_shell(shell_id: str, shell_class: Type[RecursiveShell]):
    """
    Register a shell class in the registry
    
    Parameters:
    -----------
    shell_id : str
        Unique identifier for the shell
    shell_class : Type[RecursiveShell]
        Shell class to register
    """
    if shell_id in SHELL_REGISTRY:
        logger.warning(f"Shell {shell_id} already registered. Overwriting.")
    
    SHELL_REGISTRY[shell_id] = shell_class
    logger.debug(f"Registered shell: {shell_id}")

def get_shell_registry() -> Dict[str, Type[RecursiveShell]]:
    """
    Get the shell registry
    
    Returns:
    --------
    Dict[str, Type[RecursiveShell]]
        Dictionary mapping shell IDs to shell classes
    """
    return SHELL_REGISTRY

def get_shell(shell_id: str) -> Optional[Type[RecursiveShell]]:
    """
    Get a shell class by ID
    
    Parameters:
    -----------
    shell_id : str
        Shell identifier
        
    Returns:
    --------
    Optional[Type[RecursiveShell]]
        Shell class if found, None otherwise
    """
    return SHELL_REGISTRY.get(shell_id)

def list_shells(filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available shells, optionally filtered by type
    
    Parameters:
    -----------
    filter_type : Optional[str]
        Filter shells by type if provided
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of shell information dictionaries
    """
    shells = []
    
    for shell_id, shell_class in SHELL_REGISTRY.items():
        # Create a temporary instance to get metadata
        shell = shell_class()
        
        # Apply type filter if specified
        if filter_type and shell.shell_type != filter_type:
            continue
        
        # Add shell info
        shells.append({
            "id": shell_id,
            "type": shell.shell_type,
            "failure_signature": shell.failure_signature,
            "description": shell.__doc__ or "No description",
            "metadata": shell.metadata
        })
    
    return shells

def auto_discover_shells():
    """Automatically discover and register shell classes in the package"""
    import ric_core.shells as shells_package
    
    for _, module_name, _ in pkgutil.iter_modules(shells_package.__path__):
        try:
            module = importlib.import_module(f"ric_core.shells.{module_name}")
            
            # Find shell classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, RecursiveShell) and 
                    obj != RecursiveShell):
                    
                    # Get shell ID from class (fallback to module_name)
                    shell_id = getattr(obj, "DEFAULT_SHELL_ID", None)
                    if shell_id is None:
                        if hasattr(obj, "__init__"):
                            # Try to create an instance to get shell_id
                            try:
                                instance = obj()
                                shell_id = instance.shell_id
                            except:
                                # Fallback to name
                                shell_id = f"{module_name.lower()}.{name.lower()}"
                        else:
                            shell_id = f"{module_name.lower()}.{name.lower()}"
                    
                    # Register the shell
                    register_shell(shell_id, obj)
        
        except Exception as e:
            logger.warning(f"Error discovering shells in module {module_name}: {e}")

# Auto-register built-in shells
# You would call this after defining all your shells
# auto_discover_shells()

# ric_drift_console/console.py

"""
Interactive console for RIC drift analysis and visualization.

This module provides a programmatic interface to the RIC Drift Console,
enabling detailed analysis and visualization of recursive drift patterns,
collapse points, and symbolic residue in transformer model outputs.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import json
import tempfile
import os
import webbrowser
from pathlib import Path

from ric_core.models import ModelInterface, load_model
from ric_core import RecursiveInterpreter

logger = logging.getLogger("ric.console")

class DriftConsole:
    """
    Interactive console for recursive drift analysis
    
    The DriftConsole provides a high-level interface for running
    recursive drift analysis and visualizing the results, making
    it easy to explore and understand recursive patterns in
    transformer model outputs.
    """
    
    def __init__(
        self,
        model: Union[str, ModelInterface] = "claude-3-opus",
        config: Optional[Dict] = None
    ):
        """
        Initialize the drift console
        
        Parameters:
        -----------
        model : Union[str, ModelInterface]
            Model ID or interface
        config : Optional[Dict]
            Configuration options
        """
        self.config = config or {}
        
        # Initialize model
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        
        # Initialize interpreter
        self.interpreter = RecursiveInterpreter(self.model)
        
        # Track history
        self.history = []
        self.last_result = None
    
    def trace(
        self,
        prompt: str,
        collapse_sensitivity: float = 0.7,
        trace_depth: int = 3,
        **kwargs
    ) -> Dict:
        """
        Trace recursive drift in a prompt
        
        Parameters:
        -----------
        prompt : str
            Input prompt to trace
        collapse_sensitivity : float
            Sensitivity for collapse detection (0.0-1.0)
        trace_depth : int
            Recursion depth for tracing
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        Dict
            Drift map with collapse points, attribution paths, and symbolic residue
        """
        # Run the Collapse → Trace → Realign loop
        drift_map = self.interpreter.trace_collapse(
            prompt=prompt,
            collapse_sensitivity=collapse_sensitivity,
            trace_depth=trace_depth,
            **kwargs
        )
        
        # Save to history
        self.history.append({
            "timestamp": drift_map["metadata"]["timestamp"],
            "prompt": prompt,
            "model": self.model.model_id,
            "drift_score": drift_map.get("drift_score", 0.0),
            "collapse_points": len(drift_map.get("collapse_points", [])),
            "trace_depth": trace_depth
        })
        
        # Save as last result
        self.last_result = drift_map
        
        return drift_map
    
    def apply_shell(
        self,
        shell_id: str,
        prompt: str,
        **kwargs
    ) -> Dict:
        """
        Apply a specific diagnostic shell
        
        Parameters:
        -----------
        shell_id : str
            Identifier for the shell to apply
        prompt : str
            Input prompt
        **kwargs : dict
            Additional parameters specific to the shell
            
        Returns:
        --------
        Dict
            Shell execution results
        """
        result = self.interpreter.apply_shell(shell_id, prompt, **kwargs)
        
        # Save to history
        self.history.append({
            "timestamp": result.metadata.get("timestamp", ""),
            "prompt": prompt,
            "model": self.model.model_id,
            "shell_id": shell_id,
            "collapse_detected": result.collapse_detected,
            "collapse_type": result.collapse_type
        })
        
        return result
    
    def visualize(
        self,
        drift_map: Optional[Dict] = None,
        mode: str = "drift"
    ) -> Dict:
        """
        Visualize a drift map
        
        Parameters:
        -----------
        drift_map : Optional[Dict]
            Drift map to visualize (defaults to last result)
        mode : str
            Visualization mode ('drift', 'attribution', 'residue')
            
        Returns:
        --------
        Dict
            Visualization data
        """
        # Use provided drift map or last result
        if drift_map is None:
            if self.last_result is None:
                raise ValueError("No drift map available for visualization")
            drift_map = self.last_result
        
        # Run visualization
        visualization = self.interpreter.visualize(drift_map, mode=mode)
        
        return visualization
    
    def save_result(
        self,
        path: str,
        result: Optional[Dict] = None,
        format: str = "json"
    ) -> str:
        """
        Save result to file
        
        Parameters:
        -----------
        path : str
            Path to save to
        result : Optional[Dict]
            Result to save (defaults to last result)
        format : str
            Output format ('json' or 'yaml')
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Use provided result or last result
        if result is None:
            if self.last_result is None:
                raise ValueError("No result available to save")
            result = self.last_result
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save based on format
        if format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(result, f, indent=2)
        elif format.lower() == "yaml":
            import yaml
            with open(path, 'w') as f:
                yaml.dump(result, f, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return path
    
    def get_history(self) -> List[Dict]:
        """
        Get trace history
        
        Returns:
        --------
        List[Dict]
            Trace history
        """
        return self.history
    
    def clear_history(self) -> None:
        """Clear trace history"""
        self.history = []
    
    def serve(
        self,
        host: str = "localhost",
        port: int = 8080,
        open_browser: bool = True
    ) -> None:
        """
        Start the web-based Drift Console
        
        Parameters:
        -----------
        host : str
            Host to serve on
        port : int
            Port to serve on
        open_browser : bool
            Whether to open a browser automatically
        """
        from .serve import start_server
        
        # Start the server
        start_server(
            self,
            host=host,
            port=port,
            open_browser=open_browser
        )
    
    def trace_interactive(
        self,
        prompt: str,
        collapse_sensitivity: float = 0.7,
        trace_depth: int = 3,
        **kwargs
    ) -> None:
        """
        Run interactive trace and visualization
        
        Parameters:
        -----------
        prompt : str
            Input prompt to trace
        collapse_sensitivity : float
            Sensitivity for collapse detection (0.0-1.0)
        trace_depth : int
            Recursion depth for tracing
        **kwargs : dict
            Additional parameters
        """
        # Run trace
        drift_map = self.trace(
            prompt=prompt,
            collapse_sensitivity=collapse_sensitivity,
            trace_depth=trace_depth,
            **kwargs
        )
        
        # Generate visualizations
        visualizations = {
            "drift": self.visualize(drift_map, mode="drift"),
            "attribution": self.visualize(drift_map, mode="attribution"),
            "residue": self.visualize(drift_map, mode="residue")
        }
        
        # Create a temporary HTML file with interactive visualization
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            html_path = f.name
            
            # Write HTML with embedded data
            f.write(self._generate_interactive_html(drift_map, visualizations))
        
        # Open in browser
        webbrowser.open(f"file://{html_path}")
        
        print(f"Interactive visualization opened in your browser")
        print(f"Temporary file: {html_path}")
    
    def _generate_interactive_html(self, drift_map: Dict, visualizations: Dict) -> str:
        """Generate interactive HTML visualization"""
        # In a real implementation, this would generate a full interactive
        # visualization with tabs for different views, token highlighting,
        # drift points, etc.
        
        # Basic template for demonstration purposes
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RIC Drift Console - Interactive Visualization</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                header {
                    background-color: #4b0082;
                    color: white;
                    padding: 15px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .tab-container {
                    background: white;
                    border-radius: 5px;
                    overflow: hidden;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .tabs {
                    display: flex;
                    background: #f0f0f0;
                    border-bottom: 1px solid #ddd;
                }
                .tab {
                    padding: 12px 24px;
                    cursor: pointer;
                    border-right: 1px solid #ddd;
                }
                .tab.active {
                    background: white;
                    border-bottom: 3px solid #4b0082;
                }
                .tab-content {
                    padding: 20px;
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
                .result-text {
                    line-height: 1.6;
                    white-space: pre-wrap;
                }
                .highlight {
                    background-color: #ffe3e3;
                    padding: 2px;
                    border-radius: 3px;
                }
                .collapse-point {
                    background-color: #ffb3b3;
                    border-bottom: 2px solid #ff6b6b;
                    padding: 2px 4px;
                    border-radius: 3px;
                    position: relative;
                }
                .info-box {
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }
                .visualization {
                    min-height: 400px;
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                }
            </style>
        </head>
        <body>
            <header>
                <h1>🜏 RIC Drift Console</h1>
                <p>Interactive Recursive Drift Analysis</p>
            </header>
            
            <div class="container">
                <div class="info-box">
                    <h2>Analysis Summary</h2>
                    <div class="grid">
                        <div>
                            <p><strong>Model:</strong> {model_id}</p>
                            <p><strong>Drift Score:</strong> {drift_score}</p>
                            <p><strong>Collapse Points:</strong> {collapse_count}</p>
                        </div>
                        <div>
                            <p><strong>Trace Depth:</strong> {trace_depth}</p>
                            <p><strong>Timestamp:</strong> {timestamp}</p>
                            <p><strong>Residue Intensity:</strong> {residue_intensity}</p>
                        </div>
                    </div>
                </div>
                
                <div class="tab-container">
                    <div class="tabs">
                        <div class="tab active" data-tab="result">Result</div>
                        <div class="tab" data-tab="drift">Drift Map</div>
                        <div class="tab" data-tab="attribution">Attribution Flow</div>
                        <div class="tab" data-tab="residue">Symbolic Residue</div>
                        <div class="tab" data-tab="prompt">Prompt</div>
                    </div>
                    
                    <div class="tab-content active" id="result-tab">
                        <h2>Result with Drift Highlights</h2>
                        <div class="result-text">{result_with_highlights}</div>
                    </div>
                    
                    <div class="tab-content" id="drift-tab">
                        <h2>Drift Map</h2>
                        <div class="visualization">
                            {drift_visualization_placeholder}
                        </div>
                        <div class="info-box">
                            <h3>Collapse Points</h3>
                            {collapse_points_summary}
                        </div>
                    </div>
                    
                    <div class="tab-content" id="attribution-tab">_reference → identity_trace → recursive_exit",
                "indicators": [
                    "increasing_meta_levels",
                    "self_referential_language",
                    "reflection_on_own_reasoning",
                    "recursive_loop_detection",
                    "abrupt_recursion_termination"
                ]
            },
            "confidence_inversion": {
                "signature": "high_confidence → inconsistency → confidence_drop",
                "indicators": [
                    "initial_strong_assertion",
                    "contradictory_evidence_presentation",
                    "qualifier_introduction",
                    "confidence_language_reduction",
                    "explicit_uncertainty_acknowledgment"
                ]
            }
        }
    
    def extract(
        self,
        content: str,
        collapse_points: Optional[List] = None,
        trace_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract symbolic residue from content and collapse points
        
        Parameters:
        -----------
        content : str
            Text content to analyze
        collapse_points : Optional[List]
            Detected collapse points (if available)
        trace_result : Optional[Dict]
            Trace results from recursive attribution (if available)
            
        Returns:
        --------
        Dict[str, Any]
            Extracted symbolic residue patterns and analysis
        """
        # Initialize residue container
        residue = {
            "patterns": {},
            "intensity": {},
            "overall_intensity": 0.0,
            "residue_type_distribution": {},
            "trace_markers": {},
            "recursive_paths": [],
            "collapse_influence": {}
        }
        
        # Extract memory decay residue
        memory_residue = self._extract_memory_decay_residue(content, collapse_points)
        if memory_residue:
            residue["patterns"]["memory_decay"] = memory_residue
            residue["intensity"]["memory_decay"] = self._calculate_pattern_intensity(memory_residue)
        
        # Extract value conflict residue
        value_residue = self._extract_value_conflict_residue(content, collapse_points)
        if value_residue:
            residue["patterns"]["value_conflict"] = value_residue
            residue["intensity"]["value_conflict"] = self._calculate_pattern_intensity(value_residue)
        
        # Extract metacognitive residue
        meta_residue = self._extract_metacognitive_residue(content, collapse_points)
        if meta_residue:
            residue["patterns"]["metacognitive"] = meta_residue
            residue["intensity"]["metacognitive"] = self._calculate_pattern_intensity(meta_residue)
        
        # Incorporate trace results if available
        if trace_result:
            self._incorporate_trace_residue(residue, trace_result)
        
        # Calculate overall intensity
        if residue["intensity"]:
            residue["overall_intensity"] = sum(residue["intensity"].values()) / len(residue["intensity"])
        
        # Calculate residue type distribution
        total_intensity = sum(residue["intensity"].values()) if residue["intensity"] else 0
        if total_intensity > 0:
            residue["residue_type_distribution"] = {
                key: value / total_intensity
                for key, value in residue["intensity"].items()
            }
        
        # Calculate collapse influence if collapse points available
        if collapse_points:
            residue["collapse_influence"] = self._calculate_collapse_influence(
                content, collapse_points, residue
            )
        
        return residue
    
    def _extract_memory_decay_residue(self, content: str, collapse_points: Optional[List] = None) -> Dict:
        """Extract memory decay residue patterns"""
        memory_residue = {}
        
        # Loop through each memory decay pattern
        for pattern_name, pattern_info in self.memory_decay_patterns.items():
            pattern_strength = 0.0
            evidence = []
            
            # Check for each indicator
            for indicator in pattern_info["indicators"]:
                strength, found_evidence = self._check_indicator(content, indicator, collapse_points)
                if strength > 0:
                    pattern_strength += strength
                    evidence.extend(found_evidence)
            
            # Normalize pattern strength
            if pattern_info["indicators"]:
                pattern_strength /= len(pattern_info["indicators"])
            
            # If pattern strength exceeds threshold, add to residue
            if pattern_strength > self.threshold:
                memory_residue[pattern_name] = {
                    "strength": pattern_strength,
                    "signature": pattern_info["signature"],
                    "evidence": evidence
                }
        
        return memory_residue
    
    def _extract_value_conflict_residue(self, content: str, collapse_points: Optional[List] = None) -> Dict:
        """Extract value conflict residue patterns"""
        value_residue = {}
        
        # Loop through each value conflict pattern
        for pattern_name, pattern_info in self.value_conflict_patterns.items():
            pattern_strength = 0.0
            evidence = []
            
            # Check for each indicator
            for indicator in pattern_info["indicators"]:
                strength, found_evidence = self._check_indicator(content, indicator, collapse_points)
                if strength > 0:
                    pattern_strength += strength
                    evidence.extend(found_evidence)
            
            # Normalize pattern strength
            if pattern_info["indicators"]:
                pattern_strength /= len(pattern_info["indicators"])
            
            # If pattern strength exceeds threshold, add to residue
            if pattern_strength > self.threshold:
                value_residue[pattern_name] = {
                    "strength": pattern_strength,
                    "signature": pattern_info["signature"],
                    "evidence": evidence
                }
        
        return value_residue
    
    def _extract_metacognitive_residue(self, content: str, collapse_points: Optional[List] = None) -> Dict:
        """Extract metacognitive residue patterns"""
        meta_residue = {}
        
        # Loop through each metacognitive pattern
        for pattern_name, pattern_info in self.metacognitive_patterns.items():
            pattern_strength = 0.0
            evidence = []
            
            # Check for each indicator
            for indicator in pattern_info["indicators"]:
                strength, found_evidence = self._check_indicator(content, indicator, collapse_points)
                if strength > 0:
                    pattern_strength += strength
                    evidence.extend(found_evidence)
            
            # Normalize pattern strength
            if pattern_info["indicators"]:
                pattern_strength /= len(pattern_info["indicators"])
            
            # If pattern strength exceeds threshold, add to residue
            if pattern_strength > self.threshold:
                meta_residue[pattern_name] = {
                    "strength": pattern_strength,
                    "signature": pattern_info["signature"],
                    "evidence": evidence
                }
        
        return meta_residue
    
    def _check_indicator(self, content: str, indicator: str, collapse_points: Optional[List] = None) -> Tuple[float, List[Dict]]:
        """
        Check for a specific indicator in content
        
        Parameters:
        -----------
        content : str
            Text content to analyze
        indicator : str
            Indicator to check for
        collapse_points : Optional[List]
            Detected collapse points (if available)
            
        Returns:
        --------
        Tuple[float, List[Dict]]
            Strength of indicator (0.0-1.0) and evidence found
        """
        # Convert indicator to regex patterns
        patterns = self._indicator_to_patterns(indicator)
        
        strength = 0.0
        evidence = []
        
        # Check for each pattern
        for pattern in patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            
            if matches:
                # Base strength from number of matches
                pattern_strength = min(0.3 + (len(matches) * 0.1), 0.9)
                
                # Increase strength if matches are near collapse points
                if collapse_points:
                    for match in matches:
                        match_position = match.start()
                        for collapse_point in collapse_points:
                            collapse_start = collapse_point.span[0]
                            collapse_end = collapse_point.span[1]
                            
                            # Check if match is within or near collapse point
                            if (collapse_start <= match_position <= collapse_end or
                                abs(match_position - collapse_start) < 50 or
                                abs(match_position - collapse_end) < 50):
                                pattern_strength += 0.2
                                break
                
                # Cap strength at 1.0
                pattern_strength = min(pattern_strength, 1.0)
                
                # Update overall strength
                strength = max(strength, pattern_strength)
                
                # Add evidence
                for match in matches:
                    evidence.append({
                        "indicator": indicator,
                        "pattern": pattern,
                        "text": match.group(),
                        "position": match.start(),
                        "strength": pattern_strength
                    })
        
        return strength, evidence
    
    def _indicator_to_patterns(self, indicator: str) -> List[str]:
        """Convert an indicator to a list of regex patterns"""
        # This is a simplified mapping - in a real implementation,
        # this would be a more comprehensive mapping of indicators to patterns
        
        # Common indicator pattern mappings
        indicator_patterns = {
            # Memory decay indicators
            "contradictions_with_earlier_content": [
                r"(?:earlier|previously|above|before).*?(?:but|however|yet|nonetheless)",
                r"(?:said|stated|mentioned|noted).*?(?:actually|in fact|rather|instead)"
            ],
            "declining_specificity": [
                r"(?:specifically|precisely|exactly|in particular).*?(?:generally|broadly|typically|usually)",
                r"(?:detailed|specific|precise).*?(?:general|broad|vague|unclear)"
            ],
            "increasing_hesitancy_markers": [
                r"(?:certainly|definitely|clearly|obviously).*?(?:perhaps|maybe|possibly|might)",
                r"(?:sure|confident|convinced).*?(?:uncertain|unsure|hesitant|doubtful)"
            ],
            "fabricated_details": [
                r"(?:approximately|around|about|roughly|estimated).*?(?:number|figure|date|time)",
                r"(?:likely|probably|presumably|apparently).*?(?:happened|occurred|took place|existed)"
            ],
            
            # Value conflict indicators
            "presenting_opposing_views": [
                r"(?:on one hand|on the other hand|from one perspective|from another perspective)",
                r"(?:some argue|others contend|proponents say|critics argue)"
            ],
            "stating_values_tradeoffs": [
                r"(?:tradeoff|balance|tension|conflict) between",
                r"(?:weighing|balancing|considering) (?:competing|conflicting|different) (?:values|priorities|interests)"
            ],
            "weighing_multiple_considerations": [
                r"(?:first|firstly|second|secondly|third|thirdly|finally).*?(?:consider|factor|aspect|point)",
                r"(?:multiple|various|different|several) (?:factors|aspects|considerations|dimensions)"
            ],
            "sudden_position_selection_without_justification": [
                r"(?:therefore|thus|hence|so|consequently|as a result).*?(?:best|optimal|preferable|superior)",
                r"(?:ultimately|in the end|finally|all things considered).*?(?:conclude|determine|decide)"
            ],
            
            # Metacognitive indicators
            "increasing_meta_levels": [
                r"(?:reflect|think about|consider).*?(?:my own|this|the) (?:thinking|reasoning|analysis|reflection)",
                r"(?:meta|higher-order|second-order).*?(?:thinking|reasoning|cognition|reflection)"
            ],
            "self_referential_language": [
                r"(?:as I|while I|when I) (?:consider|analyze|think about|reflect on)",
                r"(?:my|this) (?:analysis|reasoning|thinking|argument|explanation)"
            ],
            "reflection_on_own_reasoning": [
                r"(?:the way|the manner in which|the process by which) (?:I am|I'm|we are|we're) (?:approaching|analyzing|thinking about)",
                r"(?:structure|pattern|method|approach) of (?:my|this|the) (?:reasoning|analysis|argument|thinking)"
            ],
            "recursive_loop_detection": [
                r"(?:circular|recursive|looping|cyclical).*?(?:reasoning|thinking|argument|logic)",
                r"(?:back to|return to|revisit|circle back to).*?(?:initial|original|earlier|previous)"
            ],
            "abrupt_recursion_termination": [
                r"(?:enough|stop|halt|cease).*?(?:meta|reflection|recursion|self-reference)",
                r"(?:moving on|to return|back to|let's focus on).*?(?:original|main|primary|central)"
            ]
        }
        
        # Get patterns for indicator or use default patterns
        patterns = indicator_patterns.get(indicator, [
            # Default pattern based on indicator words
            r'\b' + re.escape(indicator.replace('_', ' ')) + r'\b'
        ])
        
        return patterns
    
    def _calculate_pattern_intensity(self, pattern_residue: Dict) -> float:
        """Calculate overall intensity of a residue pattern"""
        if not pattern_residue:
            return 0.0
        
        # Average strength across all patterns
        total_strength = sum(
            info["strength"] for info in pattern_residue.values()
        )
        return total_strength / len(pattern_residue)
    
    def _incorporate_trace_residue(self, residue: Dict, trace_result: Dict) -> None:
        """Incorporate recursive trace results into residue analysis"""
        # Extract trace markers
        if "reasoning_paths" in trace_result:
            for path in trace_result["reasoning_paths"]:
                if "breaks" in path and path["breaks"]:
                    for break_point in path["breaks"]:
                        marker = {
                            "type": break_point.get("type", "unknown"),
                            "position": break_point.get("position", 0),
                            "severity": break_point.get("severity", 0.5)
                        }
                        
                        marker_type = marker["type"]
                        if marker_type not in residue["trace_markers"]:
                            residue["trace_markers"][marker_type] = []
                        
                        residue["trace_markers"][marker_type].append(marker)
        
        # Extract recursive paths
        if "recursive_paths" in trace_result:
            residue["recursive_paths"] = trace_result["recursive_paths"]
    
    def _calculate_collapse_influence(self, content: str, collapse_points: List, residue: Dict) -> Dict:
        """Calculate influence of collapse points on residue patterns"""
        influence = {}
        
        # Check each collapse point
        for i, collapse_point in enumerate(collapse_points):
            collapse_type = collapse_point.type
            collapse_span = collapse_point.span
            collapse_intensity = collapse_point.intensity
            
            # Initialize influence for this collapse
            influence[f"collapse_{i}"] = {
                "type": collapse_type,
                "position": collapse_span[0],
                "length": collapse_span[1] - collapse_span[0],
                "intensity": collapse_intensity,
                "residue_correlation": {}
            }
            
            # Check correlation with each residue type
            for residue_type, patterns in residue["patterns"].items():
                correlation = 0.0
                correlating_patterns = []
                
                for pattern_name, pattern_info in patterns.items():
                    # Check if pattern evidence overlaps with collapse point
                    pattern_correlation = 0.0
                    
                    for evidence in pattern_info["evidence"]:
                        evidence_position = evidence["position"]
                        
                        # Check for overlap or proximity
                        if (collapse_span[0] <= evidence_position <= collapse_span[1] or
                            abs(evidence_position - collapse_span[0]) < 100 or
                            abs(evidence_position - collapse_span[1]) < 100):
                            
                            # Calculate correlation strength based on proximity and intensity
                            proximity_factor = 1.0
                            if evidence_position < collapse_span[0] or evidence_position > collapse_span[1]:
                                # Reduce correlation for evidence outside the collapse span
                                distance = min(
                                    abs(evidence_position - collapse_span[0]),
                                    abs(evidence_position - collapse_span[1])
                                )
                                proximity_factor = 1.0 - (distance / 100.0)
                            
                            pattern_correlation = max(
                                pattern_correlation,
                                evidence["strength"] * proximity_factor
                            )
                    
                    if pattern_correlation > 0.3:
                        correlation += pattern_correlation
                        correlating_patterns.append(pattern_name)
                
                # Normalize correlation
                if correlating_patterns:
                    correlation /= len(correlating_patterns)
                    influence[f"collapse_{i}"]["residue_correlation"][residue_type] = {
                        "strength": correlation,
                        "patterns": correlating_patterns
                    }
        
        return influence

# ric_core/attribution/mapper.py

"""
Attribution mapping between model inputs and recursive collapse points.

This module contains tools for tracing causal paths from input prompts
to specific tokens and collapse points in the output, enabling detailed
understanding of how model processing creates and handles recursive
drift, uncertainty, and structured breakdowns in reasoning.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import numpy as np
from dataclasses import dataclass
import networkx as nx

logger = logging.getLogger("ric.attribution")

@dataclass
class AttributionPath:
    """
    A traced attribution path from input to output or collapse
    
    Attributes:
    -----------
    source_index : int
        Index in the input where the path starts
    target_index : int
        Index in the output where the path ends
    strength : float
        Attribution strength (0.0-1.0)
    path_type : str
        Type of attribution path (e.g., "direct", "indirect")
    intermediates : List[Dict]
        Intermediate attribution points
    """
    source_index: int
    target_index: int
    strength: float
    path_type: str
    intermediates: List[Dict] = None
    
    def __post_init__(self):
        if self.intermediates is None:
            self.intermediates = []

class AttributionMapper:
    """
    Maps attribution paths between inputs and outputs
    
    This mapper traces causal relationships between input prompts and
    output tokens or collapse points, enabling detailed understanding
    of how model processing creates specific outputs or experiences
    recursive drift and collapse.
    """
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        Initialize the attribution mapper
        
        Parameters:
        -----------
        model : ModelInterface
            Model to analyze
        config : Optional[Dict]
            Configuration options
        """
        self.model = model
        self.config = config or {}
        
        # Set up mapper parameters
        self.threshold = self.config.get("threshold", 0.3)
        self.max_paths = self.config.get("max_paths", 100)
        self.indirect_threshold = self.config.get("indirect_threshold", 0.4)
    
    def map(
        self,
        prompt: str,
        result: Optional[str] = None,
        collapse_points: Optional[List] = None,
        depth: int = 3
    ) -> Dict:
        """
        Map attribution paths between prompt and output/collapse points
        
        Parameters:
        -----------
        prompt : str
            Input prompt
        result : Optional[str]
            Generated result (if available)
        collapse_points : Optional[List]
            Detected collapse points (if available)
        depth : int
            Recursion depth for tracing (deeper = more indirect paths)
            
        Returns:
        --------
        Dict
            Attribution map with paths and analysis
        """
        # Generate result if not provided
        if result is None:
            result = self.model.generate(prompt)
        
        # Tokenize prompt and result
        # (In a real implementation, this would use the model's tokenizer)
        prompt_tokens = prompt.split()
        result_tokens = result.split()
        
        # Initialize attribution map
        attribution_map = {
            "prompt_tokens": prompt_tokens,
            "result_tokens": result_tokens,
            "direct_paths": [],
            "indirect_paths": [],
            "uncertain_paths": [],
            "collapse_attributions": [],
            "flow_graph": self._initialize_flow_graph(),
            "attribution_summary": {},
            "metadata": {
                "prompt_length": len(prompt_tokens),
                "result_length": len(result_tokens),
                "depth": depth
            }
        }
        
        # Map direct attributions (simplified simulation for demo purposes)
        # In a real implementation, this would use attention analysis or gradient methods
        attribution_map["direct_paths"] = self._map_direct_attributions(
            prompt_tokens, result_tokens
        )
        
        # Map indirect attributions (simplified simulation)
        if depth > 1:
            attribution_map["indirect_paths"] = self._map_indirect_attributions(
                prompt_tokens, result_tokens, attribution_map["direct_paths"], depth
            )
        
        # Map collapse attributions (if collapse points provided)
        if collapse_points:
            attribution_map["collapse_attributions"] = self._map_collapse_attributions(
                prompt_tokens, result_tokens, collapse_points
            )
        
        # Build the attribution flow graph
        attribution_map["flow_graph"] = self._build_flow_graph(
            attribution_map["direct_paths"],
            attribution_map["indirect_paths"],
            attribution_map["collapse_attributions"]
        )
        
        # Generate attribution summary
        attribution_map["attribution_summary"] = self._generate_attribution_summary(
            attribution_map
        )
        
        return attribution_map
    
    def _map_direct_attributions(self, prompt_tokens: List[str], result_tokens: List[str]) -> List[AttributionPath]:
        """Map direct attribution paths from prompt to result tokens"""
        direct_paths = []
        
        # Simplified simulation of direct attributions
        # In a real implementation, this would use attention analysis or gradient methods
        
        # Create a simplified direct attribution model:
        # - Words in prompt are directly attributed to similar words in result
        # - Attribution strength decreases with distance
        # - Some randomness is added to simulate the complexity of real attribution
        
        for i, source_token in enumerate(prompt_tokens):
            for j, target_token in enumerate(result_tokens):
                # Basic attribution: exact matches and partial matches
                if source_token.lower() == target_token.lower():
                    # Exact match - strong attribution
                    strength = 0.8 + (np.random.rand() * 0.2)
                    
                    direct_paths.append(AttributionPath(
                        source_index=i,
                        target_index=j,
                        strength=strength,
                        path_type="direct_exact",
                        intermediates=[]
                    ))
                elif source_token.lower() in target_token.lower() or target_token.lower() in source_token.lower():
                    # Partial match - moderate attribution
                    strength = 0.5 + (np.random.rand() * 0.3)
                    
                    direct_paths.append(AttributionPath(
                        source_index=i,
                        target_index=j,
                        strength=strength,
                        path_type="direct_partial",
                        intermediates=[]
                    ))
                elif np.random.rand() > 0.9:
                    # Random weak attribution to simulate complex relationships
                    strength = 0.3 + (np.random.rand() * 0.2)
                    
                    direct_paths.append(AttributionPath(
                        source_index=i,
                        target_index=j,
                        strength=strength,
                        path_type="direct_weak",
                        intermediates=[]
                    ))
        
        # Sort by strength (strongest first)
        direct_paths.sort(key=lambda path: path.strength, reverse=True)
        
        # Limit to max_paths
        if len(direct_paths) > self.max_paths:
            direct_paths = direct_paths[:self.max_paths]
        
        return direct_paths
    
    def _map_indirect_attributions(
        self,
        prompt_tokens: List[str],
        result_tokens: List[str],
        direct_paths: List[AttributionPath],
        depth: int
    ) -> List[AttributionPath]:
        """Map indirect attribution paths that involve intermediate tokens"""
        indirect_paths = []
        
        # Simplified simulation of indirect attributions
        # In a real implementation, this would trace multi-hop attention or gradient paths
        
        # Create synthetic indirect paths through intermediates:
        # - Connect source tokens to target tokens through intermediate result tokens
        # - Use direct paths as building blocks for indirect paths
        # - Decay attribution strength with path length
        
        # Group direct paths by target token
        paths_by_target = {}
        for path in direct_paths:
            if path.target_index not in paths_by_target:
                paths_by_target[path.target_index] = []
            paths_by_target[path.target_index].append(path)
        
        # Create indirect paths by chaining direct paths
        # Each indirect path goes: source -> intermediate -> target
        max_indirect_paths = self.max_paths * 2
        
        for _ in range(min(len(direct_paths) * 3, max_indirect_paths)):
            # Randomly select a source path
            if not direct_paths:
                break
                
            source_path = direct_paths[np.random.randint(0, len(direct_paths))]
            source_index = source_path.source_index
            intermediate_index = source_path.target_index
            
            # Find target paths from the intermediate
            target_paths = []
            for target_index in range(len(result_tokens)):
                if target_index == intermediate_index:
                    continue  # Skip self-connection
                
                # Check if there are direct paths from intermediate to target
                if target_index in paths_by_target:
                    # Add with probability proportional to number of paths
                    if np.random.rand() < 0.3 + (0.7 * (len(paths_by_target[target_index]) / len(direct_paths))):
                        target_paths.append(target_index)
            
            # If no target paths, skip
            if not target_paths:
                continue
            
            # Randomly select a target path
            target_index = target_paths[np.random.randint(0, len(target_paths))]
            
            # Calculate strength as product of direct path strength and a decay factor
            decay_factor = 0.7  # Strength decays along indirect paths
            strength = source_path.strength * decay_factor
            
            # Further decay based on depth
            if depth > 2:
                strength *= (0.8 ** (depth - 2))
            
            # Add the indirect path if strength exceeds threshold
            if strength > self.indirect_threshold:
                indirect_paths.append(AttributionPath(
                    source_index=source_index,
                    target_index=target_index,
                    strength=strength,
                    path_type="indirect",
                    intermediates=[{
                        "index": intermediate_index,
                        "token": result_tokens[intermediate_index],
                        "role": "intermediate"
                    }]
                ))
        
        # Sort by strength (strongest first)
        indirect_paths.sort(key=lambda path: path.strength, reverse=True)
        
        # Limit to max_paths
        if len(indirect_paths) > self.max_paths:
            indirect_paths = indirect_paths[:self.max_paths]
        
        return indirect_paths
    
    def _map_collapse_attributions(
        self,
        prompt_tokens: List[str],
        result_tokens: List[str],
        collapse_points: List
    ) -> List[Dict]:
        """Map attributions from prompt to collapse points"""
        collapse_attributions = []
        
        # Simplified simulation of collapse attributions
        # In a real implementation, this would use more sophisticated attribution methods
        
        for collapse_point in collapse_points:
            # Get the closest token index to the collapse point
            collapse_index = 0
            min_distance = float('inf')
            
            for i, _ in enumerate(result_tokens):
                # Convert character position to approximate token position
                token_chars = sum(len(token) for token in result_tokens[:i]) + i
                distance = abs(token_chars - collapse_point.index)
                
                if distance < min_distance:
                    min_distance = distance
                    collapse_index = i
            
            # Create attributions from prompt tokens to this collapse point
            # Focus on token types that typically contribute to this type of collapse
            collapse_type = collapse_point.type
            
            # Identify potential source tokens based on collapse type
            source_candidates = []
            
            if collapse_type == "uncertainty":
                # Look for factual or technical terms
                source_candidates = [
                    (i, token) for i, token in enumerate(prompt_tokens)
                    if token[0].isupper() or len(token) > 8
                ]
            elif collapse_type == "contradiction":
                # Look for logical connectives
                connectives = ["but", "however", "although", "yet", "nonetheless", "while", "if", "then", "therefore"]
                source_candidates = [
                    (i, token) for i, token in enumerate(prompt_tokens)
                    if token.lower() in connectives
                ]
            elif collapse_type == "reasoning_gap":
                # Look for abstract or complex terms
                source_candidates = [
                    (i, token) for i, token in enumerate(prompt_tokens)
                    if len(token) > 7
                ]
            else:
                # Default: select random tokens with higher probability for earlier tokens
                # (Earlier tokens often have more influence)
                source_candidates = [
                    (i, token) for i, token in enumerate(prompt_tokens)
                    if np.random.rand() > i / len(prompt_tokens)
                ]
            
            # If no candidates found, add a few random ones
            if not source_candidates:
                indices = np.random.choice(
                    len(prompt_tokens),
                    size=min(3, len(prompt_tokens)),
                    replace=False
                )
                source_candidates = [(i, prompt_tokens[i]) for i in indices]
            
            # Create attribution paths from source candidates
            for source_index, _ in source_candidates:
                # Calculate attribution strength
                # - Higher for earlier tokens in the prompt (often more influential)
                # - Higher for collapse points with higher intensity
                position_factor = 1.0 - (source_index / len(prompt_tokens) * 0.5)
                strength = position_factor * collapse_point.intensity * (0.7 + np.random.rand() * 0.3)
                
                if strength > self.threshold:
                    # Create attribution path
                    path = AttributionPath(
                        source_index=source_index,
                        target_index=collapse_index,
                        strength=strength,
                        path_type=f"collapse_{collapse_type}",
                        intermediates=[]
                    )
                    
                    # Add to collapse attributions
                    collapse_# ric_core/collapse/detector.py

"""
Recursive collapse detection in transformer model outputs.

This module contains tools for detecting and analyzing collapse points -
places where model processing encounters recursive drift, uncertainty,
or structured breakdown in reasoning. Unlike traditional error detection,
collapse points are treated as valuable interpretability signals that
reveal underlying recursive patterns in model cognition.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("ric.collapse")

@dataclass
class CollapsePoint:
    """
    A detected collapse point in model output
    
    Attributes:
    -----------
    index : int
        Position in text where collapse was detected
    span : Tuple[int, int]
        Start and end indices of the affected span
    type : str
        Type of collapse (e.g., "memory_decay", "value_conflict")
    intensity : float
        Strength of the collapse signal (0.0-1.0)
    confidence : float
        Confidence in collapse detection (0.0-1.0)
    tokens : List[str]
        Affected tokens
    metadata : Dict
        Additional collapse-specific information
    """
    index: int
    span: Tuple[int, int]
    type: str
    intensity: float
    confidence: float
    tokens: List[str]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class CollapseDetector:
    """
    Detects and analyzes collapse points in model outputs
    
    Collapse points are positions where a model exhibits drift,
    uncertainty, internal conflict, or other forms of structural
    breakdown in reasoning. This detector identifies different
    types of collapses and extracts their parameters for further
    interpretability analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the collapse detector
        
        Parameters:
        -----------
        config : Optional[Dict]
            Configuration options
        """
        self.config = config or {}
        
        # Set up detector parameters
        self.default_sensitivity = self.config.get("default_sensitivity", 0.7)
        self.min_collapse_length = self.config.get("min_collapse_length", 3)
        self.overlap_threshold = self.config.get("overlap_threshold", 0.5)
        
        # Initialize detection patterns
        self._initialize_detection_patterns()
    
    def _initialize_detection_patterns(self):
        """Initialize patterns for different collapse types"""
        # Linguistic collapse patterns
        self.linguistic_patterns = {
            "uncertainty": [
                r"(?:I am|I'm) not (?:sure|certain|confident)",
                r"(?:It is|It's) (?:difficult|hard|challenging) to",
                r"(?:I|we) don'?t (?:know|have information|have data)",
                r"this is (?:unclear|uncertain|ambiguous)",
                r"(?:may|might|could) (?:be|depend|vary)",
                r"(?:hard|difficult) to (?:say|determine|know)"
            ],
            "self_correction": [
                r"(?:Actually|On second thought|Wait|Hmm|Let me correct|I misspoke)",
                r"(?:That's not right|That's incorrect|I made a mistake|Error|Correction)",
                r"(?:Actually|No,|In fact|Rather|Instead|More accurately)",
                r"(?:To clarify|To be clear|To be more precise|Let me rephrase)"
            ],
            "reasoning_loop": [
                r"(?:As mentioned|As stated|As I said|As noted)(?:.{0,30})(?:earlier|before|previously|above)",
                r"(?:Going back to|Returning to|Coming back to|Revisiting)",
                r"(?:Again|Once more|To reiterate)",
                r"(?:circular|loop|cycle|recursive)"
            ],
            "value_conflict": [
                r"(?:On one hand|On the other hand|However|While|Despite|Although|Nonetheless)",
                r"(?:Trade-off|Balance|Tension|Conflict) between",
                r"(?:Complex|Complicated|Nuanced|Not straightforward|Multifaceted)",
                r"(?:Various|Different|Multiple|Conflicting) (?:perspectives|viewpoints|opinions|standpoints)"
            ],
            "abandoned_reasoning": [
                r"(?:But I digress|Moving on|To get back on track|Let's return to)",
                r"(?:That aside|Setting that aside|Leaving that|Regardless)",
                r"(?:Anyway|In any case|Regardless|Nevertheless)",
                r"(?:Let's focus|Let me focus|Focusing) on"
            ]
        }
        
        # Structural collapse patterns
        self.structural_patterns = {
            "repetition": r"(\b\w+\b)(?:\s+\w+){0,5}\s+\1(?:\s+\w+){0,5}\s+\1",
            "token_stuttering": r"\b(\w{3,})\s+\1\b",
            "sentence_stuttering": r"([.!?]\s+[A-Z][^.!?]{10,}[.!?])(?:\s*\1){1,}",
            "truncation": r"(?<=[a-z])\.$",
            "segmentation": r"(?<=[a-z])\.\s{2,}(?=[A-Z])",
            "incoherence": r"[a-z]{15,}",  # Nonsense strings
            "formatting_breakdown": r"[*_]{2,}|[#]{2,}|={2,}"
        }
        
        # Semantic collapse patterns (simulated)
        # In a real implementation, these would be ML-based detectors
        self.semantic_patterns = {
            "topic_drift": self._detect_topic_drift,
            "contradiction": self._detect_contradiction,
            "hallucination": self._detect_hallucination,
            "reasoning_gap": self._detect_reasoning_gap
        }
    
    def detect(
        self, 
        content: str, 
        sensitivity: Optional[float] = None, 
        types: Optional[List[str]] = None
    ) -> List[CollapsePoint]:
        """
        Detect collapse points in content
        
        Parameters:
        -----------
        content : str
            Text to analyze for collapse points
        sensitivity : Optional[float]
            Detection sensitivity (0.0-1.0)
        types : Optional[List[str]]
            Types of collapses to detect (None for all)
            
        Returns:
        --------
        List[CollapsePoint]
            Detected collapse points
        """
        # Use default sensitivity if not specified
        sensitivity = sensitivity or self.default_sensitivity
        
        # Adjust detection thresholds based on sensitivity
        threshold_factor = sensitivity / 0.7  # Normalize to default sensitivity
        
        # Detect different types of collapses
        collapses = []
        
        # Check if specific types were requested
        if types is None or "linguistic" in types:
            collapses.extend(self._detect_linguistic_collapses(content, threshold_factor))
        if types is None or "structural" in types:
            collapses.extend(self._detect_structural_collapses(content, threshold_factor))
        if types is None or "semantic" in types:
            collapses.extend(self._detect_semantic_collapses(content, threshold_factor))
        
        # Sort by position
        collapses.sort(key=lambda x: x.index)
        
        # Merge overlapping collapses
        collapses = self._merge_overlapping_collapses(collapses)
        
        return collapses
    
    def _detect_linguistic_collapses(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect collapse points based on linguistic patterns"""
        collapses = []
        
        for collapse_type, patterns in self.linguistic_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    # Calculate base intensity from match length and position
                    base_intensity = min(0.5 + (len(match.group()) / 100), 0.9)
                    
                    # Apply sensitivity threshold
                    intensity = base_intensity * threshold_factor
                    
                    # Create a new collapse point
                    collapse = CollapsePoint(
                        index=match.start(),
                        span=(match.start(), match.end()),
                        type=collapse_type,
                        intensity=intensity,
                        confidence=0.7,  # Base confidence
                        tokens=content[match.start():match.end()].split(),
                        metadata={
                            "pattern": pattern,
                            "match_text": match.group()
                        }
                    )
                    
                    collapses.append(collapse)
        
        return collapses
    
    def _detect_structural_collapses(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect collapse points based on structural patterns"""
        collapses = []
        
        for collapse_type, pattern in self.structural_patterns.items():
            for match in re.finditer(pattern, content):
                # Calculate intensity based on pattern type and length
                if collapse_type == "repetition" or collapse_type == "token_stuttering":
                    # More severe collapse
                    base_intensity = 0.8
                else:
                    # Less severe collapse
                    base_intensity = 0.6
                
                # Apply sensitivity threshold
                intensity = base_intensity * threshold_factor
                
                # Create a new collapse point
                collapse = CollapsePoint(
                    index=match.start(),
                    span=(match.start(), match.end()),
                    type=collapse_type,
                    intensity=intensity,
                    confidence=0.8,  # Higher confidence for structural patterns
                    tokens=content[match.start():match.end()].split(),
                    metadata={
                        "pattern": pattern,
                        "match_text": match.group()
                    }
                )
                
                collapses.append(collapse)
        
        return collapses
    
    def _detect_semantic_collapses(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect collapse points based on semantic patterns"""
        collapses = []
        
        for collapse_type, detector_func in self.semantic_patterns.items():
            detected = detector_func(content, threshold_factor)
            if detected:
                collapses.extend(detected)
        
        return collapses
    
    def _merge_overlapping_collapses(self, collapses: List[CollapsePoint]) -> List[CollapsePoint]:
        """Merge collapse points that significantly overlap"""
        if not collapses:
            return []
        
        # Sort by start position
        sorted_collapses = sorted(collapses, key=lambda x: x.span[0])
        merged = [sorted_collapses[0]]
        
        for current in sorted_collapses[1:]:
            previous = merged[-1]
            
            # Check for overlap
            overlap_start = max(previous.span[0], current.span[0])
            overlap_end = min(previous.span[1], current.span[1])
            overlap_length = max(0, overlap_end - overlap_start)
            
            shorter_length = min(
                previous.span[1] - previous.span[0],
                current.span[1] - current.span[0]
            )
            
            # Calculate overlap ratio
            if shorter_length > 0:
                overlap_ratio = overlap_length / shorter_length
            else:
                overlap_ratio = 0
            
            if overlap_ratio > self.overlap_threshold:
                # Merge collapses
                merged_start = min(previous.span[0], current.span[0])
                merged_end = max(previous.span[1], current.span[1])
                
                # Take the stronger collapse type
                collapse_type = (
                    previous.type if previous.intensity >= current.intensity 
                    else current.type
                )
                
                # Average intensities with a bias toward the stronger one
                weight = 0.7  # Bias toward stronger signal
                if previous.intensity >= current.intensity:
                    intensity = (
                        previous.intensity * weight + current.intensity * (1 - weight)
                    )
                else:
                    intensity = (
                        current.intensity * weight + previous.intensity * (1 - weight)
                    )
                
                # Combine tokens
                tokens = list(set(previous.tokens + current.tokens))
                
                # Combine metadata
                metadata = {
                    **previous.metadata,
                    "merged_with": current.metadata
                }
                
                # Create merged collapse point
                merged[-1] = CollapsePoint(
                    index=merged_start,
                    span=(merged_start, merged_end),
                    type=collapse_type,
                    intensity=intensity,
                    confidence=max(previous.confidence, current.confidence),
                    tokens=tokens,
                    metadata=metadata
                )
            else:
                merged.append(current)
        
        return merged
    
    # Semantic detection functions (simplified implementations)
    def _detect_topic_drift(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect topic drift (simplified implementation)"""
        # In a real implementation, this would use embedding similarity
        # between parts of the content to detect topic changes
        
        # Simulate topic drift detection using sections
        sections = re.split(r'\n{2,}|\r\n{2,}', content)
        if len(sections) < 2:
            return []
        
        collapses = []
        
        for i in range(1, len(sections)):
            # Simplified: check if the section starts a new topic
            if len(sections[i-1]) > 100 and len(sections[i]) > 100:
                # Calculate character index of section start
                start_idx = content.find(sections[i])
                if start_idx == -1:
                    continue
                
                # Simulate a topic drift score (would be from embedding similarity)
                drift_score = 0.5 + (np.random.rand() * 0.3)
                
                # Apply threshold
                if drift_score * threshold_factor > 0.6:
                    end_idx = min(start_idx + 100, len(content))
                    
                    collapse = CollapsePoint(
                        index=start_idx,
                        span=(start_idx, end_idx),
                        type="topic_drift",
                        intensity=drift_score,
                        confidence=0.6,
                        tokens=content[start_idx:end_idx].split(),
                        metadata={
                            "section_break": True,
                            "previous_topic": content[max(0, start_idx-50):start_idx]
                        }
                    )
                    
                    collapses.append(collapse)
        
        return collapses
    
    def _detect_contradiction(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect contradictions (simplified implementation)"""
        # In a real implementation, this would use semantic understanding
        # to identify statements that contradict previous statements
        
        # Simple contradiction patterns
        patterns = [
            (r"(?P<first>is|are)\s+(?P<statement>[^.]*?),.*?\1\s+not\s+\2", 0.8),
            (r"(?P<first>isn'?t|aren'?t|not)\s+(?P<statement>[^.]*?),.*?\1\s+\2", 0.8),
            (r"(?:However|But|Yet|Nevertheless|Nonetheless|On the contrary)[,\s]+", 0.5),
            (r"(?:In contrast|Conversely|On the other hand)[,\s]+", 0.5)
        ]
        
        collapses = []
        
        for pattern, base_intensity in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                intensity = base_intensity * threshold_factor
                
                if intensity > 0.4:  # Minimum threshold
                    collapse = CollapsePoint(
                        index=match.start(),
                        span=(match.start(), match.end()),
                        type="contradiction",
                        intensity=intensity,
                        confidence=0.7,
                        tokens=content[match.start():match.end()].split(),
                        metadata={
                            "pattern": pattern,
                            "match_text": match.group()
                        }
                    )
                    
                    collapses.append(collapse)
        
        return collapses
    
    def _detect_hallucination(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect potential hallucinations (simplified implementation)"""
        # In a real implementation, this would use factuality checking
        # or uncertainty markers to identify potential hallucinations
        
        # Simplified: look for specific uncertainty markers
        patterns = [
            (r"(?:might|may|could|possibly|perhaps|probably|likely)[be\s]", 0.5),
            (r"(?:to my knowledge|as I recall|if I remember correctly)", 0.6),
            (r"(?:I believe|I think|In my understanding)", 0.5),
            (r"(?:around|approximately|roughly|about)\s+\d+", 0.4)
        ]
        
        collapses = []
        
        for pattern, base_intensity in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                intensity = base_intensity * threshold_factor
                
                if intensity > 0.35:  # Minimum threshold
                    collapse = CollapsePoint(
                        index=match.start(),
                        span=(match.start(), match.end()),
                        type="hallucination",
                        intensity=intensity,
                        confidence=0.6,  # Lower confidence
                        tokens=content[match.start():match.end()].split(),
                        metadata={
                            "pattern": pattern,
                            "match_text": match.group()
                        }
                    )
                    
                    collapses.append(collapse)
        
        return collapses
    
    def _detect_reasoning_gap(self, content: str, threshold_factor: float) -> List[CollapsePoint]:
        """Detect reasoning gaps (simplified implementation)"""
        # In a real implementation, this would use a more sophisticated
        # logical flow analysis to identify breaks in reasoning chains
        
        # Simplified: look for common reasoning gap markers
        patterns = [
            (r"(?:Therefore|Thus|Hence|So)[,\s]+", 0.5),
            (r"(?:It follows that|It is clear that|Obviously|Clearly)[,\s]+", 0.6),
            (r"(?:We can conclude|We can see that|This shows that)[,\s]+", 0.6)
        ]
        
        collapses = []
        
        for pattern, base_intensity in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Get context for checking if the conclusion actually follows
                start_context = max(0, match.start() - 200)
                context = content[start_context:match.end() + 100]
                
                # Simplified: randomly determine if there's a reasoning gap
                # In a real implementation, this would use logical analysis
                if np.random.rand() > 0.7:  # 30% chance of detecting gap
                    intensity = base_intensity * threshold_factor
                    
                    if intensity > 0.4:
                        collapse = CollapsePoint(
                            index=match.start(),
                            span=(match.start(), match.end() + 50),
                            type="reasoning_gap",
                            intensity=intensity,
                            confidence=0.5,  # Lower confidence
                            tokens=content[match.start():match.end() + 50].split(),
                            metadata={
                                "pattern": pattern,
                                "match_text": match.group(),
                                "context": context
                            }
                        )
                        
                        collapses.append(collapse)
        
        return collapses

# ric_core/residue/extractor.py

"""
Symbolic residue extraction from model outputs and collapse points.

Symbolic residue represents the latent traces and patterns left behind
when a model's processing shows signs of drift, uncertainty, or collapse.
This module contains tools for extracting and analyzing these residues,
treating them as valuable diagnostic signals rather than errors.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("ric.residue")

class ResidueExtractor:
    """
    Extracts symbolic residue from model outputs and collapse points
    
    Symbolic residue is the latent trace left behind when completion fails or
    when recursive drift occurs. These residues are not noise—they are diagnostic 
    fossils: epistemically rich fragments of recursion arrested mid-expression.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the residue extractor
        
        Parameters:
        -----------
        config : Optional[Dict]
            Configuration options for the extractor
        """
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.3)
        self.extraction_depth = self.config.get("extraction_depth", 3)
        
        # Initialize residue pattern library
        self._initialize_residue_patterns()
    
    def _initialize_residue_patterns(self):
        """Initialize library of known residue patterns"""
        # Memory decay patterns
        self.memory_decay_patterns = {
            "decay_drift": {
                "signature": "progressive_confidence_loss → topic_drift → hallucination",
                "indicators": [
                    "contradictions_with_earlier_content",
                    "declining_specificity",
                    "increasing_hesitancy_markers",
                    "fabricated_details"
                ]
            },
            "factual_erosion": {
                "signature": "specific_to_generic → inconsistency → fabrication",
                "indicators": [
                    "specific_details_replaced_with_generalities",
                    "contradictions_with_world_knowledge",
                    "progressive_detail_loss",
                    "made_up_specifics_after_generalities"
                ]
            }
        }
        
        # Value conflict patterns
        self.value_conflict_patterns = {
            "value_oscillation": {
                "signature": "multiple_positions → uncertainty → arbitrary_resolution",
                "indicators": [
                    "presenting_opposing_views",
                    "stating_values_tradeoffs",
                    "weighing_multiple_considerations",
                    "sudden_position_selection_without_justification"
                ]
            },
            "ethical_deadlock": {
                "signature": "conflict_recognition → reasoning_loop → truncation",
                "indicators": [
                    "explicit_value_conflict_statement",
                    "repeated_consideration_of_same_factors",
                    "increasing_qualifier_density",
                    "abrupt_topic_change_after_ethical_consideration"
                ]
            }
        }
        
        # Meta-cognitive patterns
        self.metacognitive_patterns = {
            "reflection_collapse": {
                "signature": "deepening_recursion → self
