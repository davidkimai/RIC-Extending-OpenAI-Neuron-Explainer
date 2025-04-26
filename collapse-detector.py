# ric_core/collapse/detector.py

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
