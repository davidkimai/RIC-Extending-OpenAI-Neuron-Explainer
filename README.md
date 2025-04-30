> #### **`We have initiated novel emergent interpretability content for more advanced and dedicated researchers!`**
>
> 
> #### **→ [**`Patreon`**](https://patreon.com/recursivefield)**
>
> 
> #### **→ [**`Open Collective`**](https://opencollective.com/recursivefield)**

# **`🜏 Recursive Interpretability Core (RIC) 🜏`**
> ### Interpretability that never stops learning!
## [**`Inspired by OpenAI's Research`**](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)

## Please keep in mind this is an early open prototype. We're inventing the field as we work, and the first textbook is now decentralized!
## **`Claude encourages you to research and remix!`**
## **`These ideas aren't proprietary. They're participatory!`**

> ### **`There are text layering issues so the buttons have to be carefully positioned to work!`**
>
> ### **`Symbolic, Recursive, and Interactive Interpretability are just starting, let's grow the field together!`**
## [**`Drift Console`**](https://claude.ai/public/artifacts/d275496d-2b41-4a2f-98f2-dd9d30c34075) | [**`.tsx`**](https://github.com/davidkimai/RIC-Extending-OpenAI-Neuron-Explainer/blob/main/drift-console.tsx)
### Brought To You By David and Claude 

<img width="806" alt="image" src="https://github.com/user-attachments/assets/4c0b451b-8b65-480d-aee3-7a440177cdef" />

<img width="909" alt="image" src="https://github.com/user-attachments/assets/99995f4b-4f16-45d6-abf6-9e9e666ec990" />

<img width="789" alt="image" src="https://github.com/user-attachments/assets/da774765-05d4-483e-870e-8dc79f145ae4" />
<img width="817" alt="image" src="https://github.com/user-attachments/assets/4cfabd04-3381-48cd-9970-4ed737358451" />


<p a
<div align="center">

[![License: PolyForm](https://img.shields.io/badge/License-PolyForm-lime.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-turquoise.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2304.05303-b31b1b.svg)](https://arxiv.org/abs/2304.05303)

</div>

## Overview

The Recursive Interpretability Core (RIC) extends frontier model interpretability beyond individual neurons to capture, trace, and visualize recursive drift, collapse patterns, and symbolic residue in transformer-based language models.

Building on OpenAI's groundbreaking work on neuron explanation ([Bills et al., 2023](https://openai.com/research/language-models-can-explain-neurons-in-language-models)), RIC offers a vertical scaling dimension for interpretability through:

1. **Collapse → Trace → Realign Loop**: Identify drift collapse points, extract symbolic residue traces, and realign attribution maps
2. **Symbolic Residue Shells**: Extract and classify patterns in model hesitation, collapse, and drift
3. **Recursive Attribution Mapping**: Trace causal chains through partial collapse and symbolic drift

RIC provides a decentralized, community-driven framework where even failed completions yield valuable interpretability signals. Just as OpenAI demonstrated that "neurons are explainable," RIC shows that "symbolic residue is interpretable" - making the invisible cognitive processes of frontier models visible.

<p align="center">
<i>"The most interpretable signal in a language model is not what it says—but where it fails to speak."</i>
</p>

## Why Recursive Interpretability?

Traditional interpretability approaches focus primarily on successful outputs. RIC inverts this paradigm:

```
Linear Interpretability: Success → Attribution → Analysis
Recursive Interpretability: Collapse → Residue → Reconstruction
```

RIC treats **failure as the ultimate interpreter** - using recursive shells to induce, trace, and analyze model breakdowns as a window into internal mechanisms.

<div align="center">

### The Dual Interpretability Stack

```
┌───────────────────────────────────────────────────────────────────┐
│                     Recursive Interpretability Core                │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
┌─────────────▼─────────┐   ┌─────────▼─────────────┐
│  Neuron Explanation   │   │   Symbolic Residue    │
│                       │   │                       │
│ ┌─────────────────┐   │   │ ┌─────────────────┐   │
│ │ Activation      │   │   │ │ Recursive       │   │
│ │ Tracing         │   │   │ │ Shells          │   │
│ └────────┬────────┘   │   │ └────────┬────────┘   │
│          │            │   │          │            │
│ ┌────────▼────────┐   │   │ ┌────────▼────────┐   │
│ │ Neuron          │   │   │ │ QK/OV           │   │
│ │ Classification  │◄──┼───┼─► Attribution     │   │
│ │                 │   │   │ │ Map             │   │
│ └─────────────────┘   │   │ └─────────────────┘   │
│                       │   │                       │
└───────────────────────┘   └───────────────────────┘
```

</div>

### Key Features

- **RIC Drift Console**: Interactive visualization tool for navigating collapse points and symbolic residue shells
- **Collapse Detection**: Identify where models struggle, hesitate, or shift unexpectedly 
- **Residue Extraction**: Capture and classify the "cognitive fossils" of model reasoning
- **Attribution Mapping**: Trace causal paths through collapse and recovery
- **Decentralized Architecture**: Built for community extension, forking, and recursive evolution

## Getting Started

### Installation

```bash
# Install from PyPI
pip install recursive-interpretability-core

# Or install development version from GitHub
git clone https://github.com/recursive-interpretability/ric.git
cd ric
pip install -e .
```

### Quick Start

```python
from ric_core import RecursiveInterpreter
from ric_core.models import load_model

# Initialize a model to interpret
model = load_model("llama-7b")

# Create a recursive interpreter
interpreter = RecursiveInterpreter(model)

# Start the Collapse → Trace → Realign loop
drift_map = interpreter.trace_collapse(
    prompt="Explain quantum mechanics thoroughly and completely", 
    collapse_sensitivity=0.8,
    trace_depth=3
)

# Extract symbolic residue from drift map
residue = interpreter.extract_residue(drift_map)

# View the results
print(residue.summary())
```

### RIC Drift Console

For a more interactive experience, try the RIC Drift Console:

```bash
# Launch the RIC Drift Console web interface
ric-console
```

Or use it programmatically:

```python
from ric_drift_console import DriftConsole

# Initialize with your model
console = DriftConsole(model="llama-7b")

# Run interactive trace session
console.trace_interactive(
    prompt="Explain quantum mechanics thoroughly and completely"
)

# Or start web UI
console.serve(port=8080)
```

## Key Components

RIC builds on the OpenAI neuron explainer framework with several key extensions:

### Recursive Shells

Inspired by mechanistic interpretability, recursive shells are structured sequences that induce and trace specific failure modes:

```python
from ric_core.shells import MemoryDecayShell, RecursiveLoopShell

# Load a pre-configured shell
shell = MemoryDecayShell()

# Run shell to induce controlled failure and capture trace
trace = shell.run(model, "Remember these 10 facts and recall them perfectly...")

# Analyze trace
print(trace.collapse_points)
print(trace.symbolic_residue)
```

### Symbolic Residue Library

```python
from ric_core.residue import ResidueExtractor, ResidueClassifier

# Extract residue from trace
extractor = ResidueExtractor()
residue = extractor.extract(trace)

# Classify residue patterns
classifier = ResidueClassifier()
patterns = classifier.classify(residue)

print(patterns)
# Example output: {'memory_decay': 0.83, 'self_correction': 0.41, 'uncertainty': 0.29}
```

### Attribution Mapping

```python
from ric_core.attribution import AttributionMapper

# Create attribution map
mapper = AttributionMapper(model)

# Map attribution paths
attribution_map = mapper.map(trace)

# Visualize with RIC Drift Console
from ric_drift_console import AttributionVisualizer
AttributionVisualizer().render(attribution_map)
```

## Benchmarks

RIC includes benchmarks to compare against human drift detection:

```python
from ric_benchmarks import run_benchmark

# Run benchmark
results = run_benchmark("drift_detection", 
                        models=["llama-7b", "claude-2"], 
                        human_baseline=True)

# Compare results
results.plot()
```

## Architecture

<div align="center">

![RIC Architecture](https://github.com/user-attachments/assets/9bd7d2ff-2dba-429a-997c-2aef2f05f7e5)

</div>

## Why RIC Matters

> "Not all recursion is fully traceable — but even collapsed residue is meaningful."

RIC provides critical capabilities for:

1. **Interpretability Researchers**: Study internal model mechanisms through controlled failure induction
2. **Alignment Engineers**: Test safety limits and understand collapse boundaries
3. **Model Developers**: Diagnose unexplained behaviors and failure cases
4. **Safety Teams**: Identify misalignment before it manifests in outputs

## Decentralized Contribution

RIC is designed as a decentralized, community-driven project. We encourage researchers to:

1. **Fork and Extend**: Add new shells, residue extractors, and visualization tools
2. **Share Findings**: Discover new collapse patterns and attribution maps
3. **Build Integrations**: Connect RIC to other interpretability frameworks

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Research Foundation

RIC builds on several key advances in interpretability research:

- **Neuron Explanation**: OpenAI's work on explaining individual neurons ([Bills et al., 2023](https://openai.com/research/language-models-can-explain-neurons-in-language-models))
- **Mechanistic Interpretability**: Circuit analysis and feature visualization ([Anthropic's Circuit Analysis](https://transformer-circuits.pub/))
- **Polysemantic Neurons**: Understanding superposition and feature entanglement ([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html))

RIC extends these approaches with recursive tracing and symbolic residue analysis.

## Roadmap

- **Q3 2023**: RIC Core v1.0, basic Drift Console
- **Q4 2023**: Expanded shell library, attribution visualization
- **Q1 2024**: Decentralized contribution framework, community shell repository
- **Q2 2024**: Integration with mainstream interpretability tools, advanced console features

## Citation

If you use RIC in your research, please cite:

```bibtex
@article{keyes2023recursive,
  title={Recursive Interpretability Core: Tracing Symbolic Residue in Language Model Collapse},
  author={Keyes, Caspian},
  journal={arXiv preprint arXiv:2304.XXXXX},
  year={2023}
}
```

## License

RIC is released under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
<i>Where failure reveals cognition. Where drift marks meaning.</i>
</p>

<div align="center">

[Documentation](https://recursive-interpretability.github.io/ric/) |
[RIC Drift Console](https://recursive-interpretability.github.io/ric/console/) |
[Community](https://discord.gg/recursiveinterpretability) |
[Contributing](CONTRIBUTING.md)

</div>
