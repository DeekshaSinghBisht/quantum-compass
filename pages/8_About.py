"""About Quantum Compass — methodology, citations and reuse guidance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

st.set_page_config(page_title="About · Quantum Compass", page_icon="📚", layout="wide")

def _github_repository_url() -> str:
    """Read an optional source URL from an environment variable or Streamlit secret."""
    environment_value = os.environ.get("QUANTUM_COMPASS_GITHUB_URL", "").strip()
    if environment_value:
        return environment_value
    try:
        return str(st.secrets.get("QUANTUM_COMPASS_GITHUB_URL", "")).strip()
    except Exception:
        return ""


_GITHUB_URL = _github_repository_url()

_FEATURED_BIBTEX = {
    "QAOA foundations": """@article{farhi2014qaoa,
  title={A Quantum Approximate Optimization Algorithm},
  author={Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam},
  journal={arXiv preprint arXiv:1411.4028},
  year={2014}
}""",
    "Radiation-induced qubit errors": """@article{vepsalainen2020radiation,
  title={Impact of ionizing radiation on superconducting qubit coherence},
  author={Vepsalainen, Antti and others},
  journal={Nature}, volume={584}, pages={551--556}, year={2020}
}""",
    "Satellite-to-ground quantum link": """@article{liao2017satellite,
  title={Satellite-to-ground quantum key distribution},
  author={Liao, Sheng-Kai and others},
  journal={Nature}, volume={549}, pages={43--47}, year={2017}
}""",
}

_SUPPORTING_BIBTEX = {
    "Goemans--Williamson MaxCut": """@article{goemans1995improved,
  title={Improved approximation algorithms for maximum cut and satisfiability problems},
  author={Goemans, Michel X. and Williamson, David P.},
  journal={Journal of the ACM}, volume={42}, number={6}, pages={1115--1145}, year={1995}
}""",
    "Error mitigation": """@article{temme2017error,
  title={Error mitigation for short-depth quantum circuits},
  author={Temme, Kristan and Bravyi, Sergey and Gambetta, Jay M.},
  journal={Physical Review Letters}, volume={119}, pages={180509}, year={2017}
}""",
    "Hardware-efficient variational methods": """@article{kandala2017hardware,
  title={Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets},
  author={Kandala, Abhinav and others},
  journal={Nature}, volume={549}, pages={242--246}, year={2017}
}""",
    "Van Allen environment": """@article{baker2013vanallen,
  title={The {V}an {A}llen Probes Mission},
  author={Baker, Daniel N. and others},
  journal={Space Science Reviews}, volume={179}, pages={531--560}, year={2013}
}""",
    "Deep-space radiation": """@article{cucinotta2014space,
  title={Space radiation risks to the central nervous system},
  author={Cucinotta, Francis A. and Kim, Myung-Hee Y. and Chappell, Lori J.},
  journal={Neuroscience and Biobehavioral Reviews}, volume={38}, pages={16--34}, year={2014}
}""",
    "Stochastic resonance": """@article{gammaitoni1998stochastic,
  title={Stochastic resonance},
  author={Gammaitoni, Luca and Hanggi, Peter and Jung, Peter and Marchesoni, Fabio},
  journal={Reviews of Modern Physics}, volume={70}, pages={223--287}, year={1998}
}""",
    "IBM Runtime fake backends": """@misc{qiskit_ibm_runtime,
  title={Qiskit IBM Runtime and Fake Provider},
  author={{IBM Quantum}},
  howpublished={Software documentation},
  year={2026},
  url={https://docs.quantum.ibm.com/}
}""",
}

_TOOL_CITATION = """@software{quantum_compass,
  title={Quantum Compass: QAOA Optimization Landscape Navigator},
  author={{Quantum Compass Contributors}},
  year={2026},
  url={https://quantum-compass.streamlit.app},
  note={Interactive QAOA, noise-environment, hardware-profile, and mission-planning toolkit}
}"""


def main() -> None:
    st.title("📚 About Quantum Compass")
    st.caption("Methodology, environment models, references and citation guidance for the QAOA landscape navigator.")

    st.subheader("What is QAOA?")
    st.write(
        "The Quantum Approximate Optimization Algorithm, or QAOA, is a hybrid quantum-classical method for "
        "combinatorial optimization. A classical optimizer chooses variational angles, while a quantum circuit "
        "alternates a problem Hamiltonian with a mixing Hamiltonian. For MaxCut, each qubit represents a graph "
        "vertex and the circuit is scored by the expected number of cut edges. Increasing depth can improve "
        "expressivity, but it also adds gates, sensitivity to calibration error, and optimization difficulty. "
        "Quantum Compass visualizes the two-dimensional gamma-beta slice used to navigate those choices, then "
        "compares ideal, terrestrial, space, and hardware-calibrated noise models so users can distinguish a "
        "promising mathematical optimum from one that remains reachable on a realistic device."
    )

    st.subheader("What is stochastic resonance?")
    st.write(
        "Stochastic resonance describes a counterintuitive situation in which a carefully matched amount of noise "
        "makes a nonlinear system's weak structure easier to detect. It is not a general claim that noise improves "
        "quantum algorithms. In Quantum Compass, competing monotone-decay and interior-peak models are fitted to "
        "a QAOA noise sweep and compared with information criteria. A resonance verdict requires an interior peak, "
        "a meaningful lift above the noiseless baseline, and a BIC advantage over simple decay. Most realistic "
        "gate-noise simulations instead show monotonic degradation. The tool reports that negative result plainly, "
        "while retaining the detector for diagnostics, synthetic controls, and future correlated-noise experiments."
    )

    st.subheader("How noise models work")
    st.write(
        "A single noise-level control is mapped to physically interpretable channels: depolarizing error on native "
        "single- and two-qubit gates, symmetric readout error, cosmic-ray burst mixtures, thermal modulation, or "
        "amplitude damping and dephasing for an optical satellite link. Exact landscapes are evaluated with Aer "
        "statevectors when ideal and density matrices when noisy. Qiskit object construction and Aer execution run "
        "inside a persistent worker subprocess, which keeps Streamlit reruns stable and prevents visual interfaces "
        "from accidentally presenting a noise-free result as noisy."
    )

    st.subheader("Space environments explained")
    environments = [
        ("Ideal", "No channels are applied; this is the exact statevector reference.", "Farhi, Goldstone & Gutmann (2014), arXiv:1411.4028."),
        ("Earth laboratory", "Representative gate depolarization and readout error model a contemporary NISQ processor.", "IBM Quantum / Quantinuum public calibration benchmarks (2023--2024)."),
        ("LEO satellite", "Elevated depolarization, cosmic-ray bursts and 90-minute thermal cycling represent a ~400 km orbit.", "Vepsäläinen et al., Nature 584, 551--556 (2020)."),
        ("Satellite-to-ground link", "Elevation-dependent photon loss, dephasing and thermal photons model an atmospheric quantum link.", "Liao et al., Nature 549, 43--47 (2017)."),
        ("MEO / Van Allen belts", "Sustained charged-particle exposure raises depolarization and burst rates.", "Baker et al., Space Science Reviews 179, 531--560 (2013)."),
        ("Deep space", "Unshielded galactic cosmic rays and solar-particle events define the strongest default profile.", "Cucinotta, Kim & Chappell, Neurosci. Biobehav. Rev. 38, 16--34 (2014)."),
        ("Custom", "User-supplied absolute channel probabilities support sensitivity analyses and calibration studies.", "Model assumptions are supplied by the user."),
    ]
    st.dataframe(
        {"Environment": [row[0] for row in environments], "Model": [row[1] for row in environments],
         "Reference": [row[2] for row in environments]},
        width="stretch", hide_index=True,
    )

    st.subheader("Methodology")
    st.markdown(
        "- **Graphs and ground truth:** seeded NetworkX graph families; MaxCut is enumerated exactly for `n ≤ 16`.\n"
        "- **Landscapes:** a gamma-beta grid evaluates deterministic approximation ratios; deeper QAOA is shown as a symmetric shared-angle slice.\n"
        "- **Worker isolation:** Qiskit/Aer noise models, FakeProvider hardware objects and density-matrix simulations are constructed only in a persistent child process.\n"
        "- **Large diagrams:** expensive maps use explicit fixed-angle or contrast-transfer protocols and annotate provenance instead of silently simulating infeasible full density matrices.\n"
        "- **Fingerprinting:** a RandomForest learns variance, extrema, gradient, roughness, symmetry and distributional features from simulated environment landscapes.\n"
        "- **Experimental validation:** a checksum-protected manifest fixes circuits, execution order and metrics before a credential-gated IBM Runtime study; uploaded measured counts are compared with named simulator models using bootstrap and distributional metrics."
    )

    st.subheader("Experimental evidence and claim boundaries")
    st.write(
        "A simulator can guide an experiment but cannot substitute for one. The Experimental Validation page therefore "
        "requires actual uploaded measurement counts before presenting a hardware comparison. Its checksum-identified manifest "
        "records the logical graph, QAOA angles, shot budget, randomized run order, bit convention and planned endpoints before "
        "job submission; formal preregistration additionally requires committing that manifest to a dated repository or archive. "
        "Reports quantify finite-shot uncertainty and model discrepancy, while explicitly excluding stronger "
        "claims about physical noise attribution, orbital environments, general quantum advantage or flight readiness."
    )

    st.subheader("Research papers and references")
    st.caption("The first three are the principal research strands supported by the tool; the following references document its optimization, mitigation, space, resonance and software context.")
    c1, c2, c3 = st.columns(3)
    for column, (title, bibtex) in zip((c1, c2, c3), _FEATURED_BIBTEX.items()):
        with column:
            st.markdown(f"**{title}**")
            st.code(bibtex, language="bibtex")
    with st.expander("Seven supporting references — BibTeX", expanded=False):
        for title, bibtex in _SUPPORTING_BIBTEX.items():
            st.markdown(f"**{title}**")
            st.code(bibtex, language="bibtex")

    st.subheader("How to cite this tool")
    st.code(_TOOL_CITATION, language="bibtex")

    st.subheader("Team, source and reuse")
    st.write(
        "Quantum Compass is an open, reproducible educational and research-prototyping project. "
        "Please cite the tool, QAOA foundations, and the relevant environment/hardware source when using its figures or derived results."
    )
    if _GITHUB_URL:
        st.link_button("Open the project GitHub repository", _GITHUB_URL, width="content")
    else:
        st.info(
            "Set the `QUANTUM_COMPASS_GITHUB_URL` environment variable after pushing the repository to expose the project link here. "
            "The deployment checklist in the README covers the GitHub and Streamlit Community Cloud steps."
        )
    st.caption("License: MIT. Author attribution: Quantum Compass Contributors; adapt the citation metadata to your institutional authorship before publication.")


main()
