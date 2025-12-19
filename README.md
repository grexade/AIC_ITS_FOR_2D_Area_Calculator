# 2D Shapes Area Tutor (Ontology-driven ITS)

A lightweight Intelligent Tutoring System (ITS) for practising **area of 2D shapes** using an OWL ontology as the **Domain Model**.

## Features
- Loads concepts (e.g., `RectangleConcept`, `CircleConcept`) dynamically from the ontology (Protégé OWL export).
- Shows ontology formulas as explainable hints (`hasFormula` → `formulaText`).
- Dip-test to initialise learner level (Novice / Intermediate / Advanced).
- Visual canvas drawings for Rectangle, Triangle, Circle, Parallelogram.
- Rule-based diagnosis of common misconceptions (perimeter vs area, missing 1/2, radius vs diameter).
- Tracks learner mastery (attempts, accuracy, average response time).

## Quick start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the ontology file is named:
`area2d_ontology.owl`

3. Run:
```bash
python area_tutor_app.py
```

## Ontology namespace
The app expects the ontology namespace:
`http://www.semanticweb.org/user/ontologies/2025/10/area2d_ontology.owl#`

If you change the namespace in Protégé, update it in `OntologyManager`.
