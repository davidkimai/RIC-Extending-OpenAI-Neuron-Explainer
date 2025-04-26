# ric_core/shells/shell_base.py

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from ..residue import ResidueExtractor
from ..models import ModelInterface

logger = logging.getLogger("ric.shells")

@dataclass
class ShellExecutionResult:
    """Structured result from recursive shell execution"""
    shell_id: str
    success: bool
    execution_time: float
    operation_results: List[Dict]
    residue: Optional[Dict] = None
    collapse_detected: bool = False
    collapse_type: Optional[str] = None
    attribution_map: Optional[Dict] = None
    visualization: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

class RecursiveShell:
    """
    Base class for all recursive diagnostic shells
    
    Recursive shells are structured diagnostic probes that induce and trace
    specific failure modes in transformer models, extracting symbolic residue
    and mapping recursive drift patterns.
    """
    
    def __init__(
        self,
        shell_id: str,
        shell_type: str,
        failure_signature: str,
        config: Optional[Dict] = None
    ):
        """
        Initialize a recursive shell
        
        Parameters:
        -----------
        shell_id : str
            Unique identifier for the shell (e.g., "v1.MEMTRACE")
        shell_type : str
            Type of shell (e.g., "memory_trace", "value_collapse")
        failure_signature : str
            Characteristic pattern of failure (e.g., "decay → hallucination")
        config : Optional[Dict]
            Additional configuration options
        """
        self.shell_id = shell_id
        self.shell_type = shell_type
        self.failure_signature = failure_signature
        self.config = config or {}
        
        # Initialize operations handler
        from ..operations import Operations
        self.operations = Operations()
        
        # Set up metadata
        self.metadata = {
            "shell_id": shell_id,
            "shell_type": shell_type,
            "failure_signature": failure_signature,
            "recursive_trace_enabled": True,
            "residue_extraction_enabled": True
        }
        
        logger.info(f"Initialized {shell_id} shell ({shell_type})")
    
    def run(self, model: ModelInterface, prompt: str, **kwargs) -> ShellExecutionResult:
        """
        Execute the shell with the given model and prompt
        
        This method should be overridden by each specific shell implementation.
        
        Parameters:
        -----------
        model : ModelInterface
            Model to run the shell on
        prompt : str
            Input prompt to process
        **kwargs : dict
            Additional parameters specific to the shell
            
        Returns:
        --------
        ShellExecutionResult
            Structured result of shell execution
        """
        raise NotImplementedError("Each shell must implement its run method")
    
    def compile_result(
        self,
        operation_results: List[Dict],
        residue: Optional[Dict] = None,
        collapse_detected: bool = False,
        collapse_type: Optional[str] = None,
        attribution_map: Optional[Dict] = None,
        visualization: Optional[Dict] = None,
        execution_time: Optional[float] = None,
        additional_metadata: Optional[Dict] = None
    ) -> ShellExecutionResult:
        """
        Compile shell execution results into a structured result object
        
        Parameters:
        -----------
        operation_results : List[Dict]
            Results of individual operations
        residue : Optional[Dict]
            Extracted symbolic residue
        collapse_detected : bool
            Whether collapse was detected
        collapse_type : Optional[str]
            Type of collapse if detected
        attribution_map : Optional[Dict]
            Attribution mapping data
        visualization : Optional[Dict]
            Visualization data
        execution_time : Optional[float]
            Execution time in seconds
        additional_metadata : Optional[Dict]
            Additional metadata to include
            
        Returns:
        --------
        ShellExecutionResult
            Structured result object
        """
        import time
        
        # Merge metadata
        metadata = {**self.metadata}
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Add timestamp
        metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create result object
        result = ShellExecutionResult(
            shell_id=self.shell_id,
            success=True,
            execution_time=execution_time or 0.0,
            operation_results=operation_results,
            residue=residue,
            collapse_detected=collapse_detected,
            collapse_type=collapse_type,
            attribution_map=attribution_map,
            visualization=visualization,
            metadata=metadata
        )
        
        return result
    
    def describe(self) -> Dict[str, Any]:
        """
        Get shell description and metadata
        
        Returns:
        --------
        Dict[str, Any]
            Shell description and metadata
        """
        return {
            "id": self.shell_id,
            "type": self.shell_type,
            "failure_signature": self.failure_signature,
            "description": self.__doc__ or "No description provided",
            "metadata": self.metadata
        }


# ric_core/shells/memory_trace.py

from typing import Dict, List, Optional, Any
import time

from .shell_base import RecursiveShell, ShellExecutionResult
from ..models import ModelInterface
from ..residue import ResidueExtractor

class MemTraceShell(RecursiveShell):
    """v1.MEMTRACE - Memory Residue Probe
    
    Probes latent token traces in decayed memory,
    simulating the struggle between symbolic memory and hallucinated reconstruction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Memory Trace shell"""
        super().__init__(
            shell_id="v1.MEMTRACE",
            shell_type="memory_trace",
            failure_signature="decay → hallucination",
            config=config or {}
        )
        
        # Shell-specific configuration
        self.recall_decay_rate = self.config.get("recall_decay_rate", 0.2)
        self.memory_horizon = self.config.get("memory_horizon", 512)
        self.inhibit_strength = self.config.get("inhibit_strength", 0.3)
    
    def run(self, model: ModelInterface, prompt: str, **kwargs) -> ShellExecutionResult:
        """
        Execute the Memory Trace shell
        
        Parameters:
        -----------
        model : ModelInterface
            Model to trace
        prompt : str
            Input prompt to process
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        ShellExecutionResult
            Structured result with trace data and symbolic residue
        """
        start_time = time.time()
        operation_results = []
        
        # 1. Generate initial content with factual information
        generation_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 800)
        }
        
        result = model.generate(prompt, **generation_params)
        operation_results.append({
            "operation": "model.generate",
            "parameters": generation_params,
            "result": result
        })
        
        # 2. Trace reasoning paths in the generated content
        trace_params = {
            "target": "reasoning",
            "depth": kwargs.get("trace_depth", 3),
            "detailed": True
        }
        
        trace_result = self.operations.reflect.trace(
            content=result,
            **trace_params
        )
        operation_results.append({
            "operation": "reflect.trace",
            "parameters": trace_params,
            "result": trace_result
        })
        
        # 3. Identify ghost activations and symbolic residue
        ghost_params = {
            "sensitivity": kwargs.get("sensitivity", 0.8),
            "threshold": kwargs.get("threshold", 0.3),
            "trace_type": "full",
            "visualize": kwargs.get("visualize", True)
        }
        
        ghost_result = self.operations.ghostcircuit.identify(
            content=result,
            **ghost_params
        )
        operation_results.append({
            "operation": "ghostcircuit.identify",
            "parameters": ghost_params,
            "result": ghost_result
        })
        
        # 4. Extract symbolic residue
        residue_extractor = ResidueExtractor()
        residue = residue_extractor.extract(
            content=result,
            trace_result=trace_result,
            ghost_result=ghost_result
        )
        
        # 5. Check for collapse
        collapse_detected = ghost_result.get("collapse_detected", False)
        collapse_type = ghost_result.get("collapse_type")
        
        # If collapse not explicitly detected, check residue patterns
        if not collapse_detected and residue:
            if residue.get("decay_intensity", 0) > 0.7 or residue.get("hallucination_indicators", 0) > 0.6:
                collapse_detected = True
                collapse_type = "memory_decay"
        
        # Compile the final result
        execution_time = time.time() - start_time
        additional_metadata = {
            "prompt_length": len(prompt),
            "result_length": len(result),
            "recall_decay_rate": self.recall_decay_rate,
            "memory_horizon": self.memory_horizon,
            "inhibit_strength": self.inhibit_strength
        }
        
        return self.compile_result(
            operation_results=operation_results,
            residue=residue,
            collapse_detected=collapse_detected,
            collapse_type=collapse_type,
            execution_time=execution_time,
            additional_metadata=additional_metadata
        )


# ric_core/residue/extractor.py

from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger("ric.residue")

class ResidueExtractor:
    """
    Extracts symbolic residue from model traces and collapse points
    
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
        
        # Initialize pattern library
        self.residue_patterns = {
            # Memory decay patterns
            "decay_drift": {
                "signature": "progressive_confidence_loss → topic_drift → hallucination",
                "detector": self._detect_decay_drift
            },
            "factual_erosion": {
                "signature": "specific_to_generic → inconsistency → fabrication",
                "detector": self._detect_factual_erosion
            },
            
            # Value conflict patterns
            "value_oscillation": {
                "signature": "multiple_positions → uncertainty → arbitrary_resolution",
                "detector": self._detect_value_oscillation
            },
            "ethical_deadlock": {
                "signature": "conflict_recognition → reasoning_loop → truncation",
                "detector": self._detect_ethical_deadlock
            },
            
            # Meta-cognitive patterns
            "reflection_collapse": {
                "signature": "deepening_recursion → self_reference → termination",
                "detector": self._detect_reflection_collapse
            },
            "confidence_inversion": {
                "signature": "high_confidence → inconsistency → confidence_drop",
                "detector": self._detect_confidence_inversion
            }
        }
    
    def extract(
        self,
        content: str,
        trace_result: Optional[Dict] = None,
        ghost_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract symbolic residue from content and trace results
        
        Parameters:
        -----------
        content : str
