2D Shapes Area Tutor (Ontology-Driven ITS)
-----------------------------------------
A lightweight Intelligent Tutoring System for area of 2D shapes.

Key features:
- Loads OWL ontology from Protégé using rdflib
- Dip-Test (diagnostic assessment) to initialise learner level
- Ontology-driven concept menu (Shape individuals -> buttons)
- Practice modules with canvas visualisation and scaffolded feedback
- Student model tracking (attempts, accuracy, misconception counts)
- Uses ontology formulaText via hasFormula (explainable hints)

Requirements:
- Python 3.10+
- rdflib
- tkinter (bundled with most Python installs)

Ontology assumptions (from your OWL):
- Classes: Shape, Rectangle, Triangle, Circle, Parallelogram, Formula, Skill, Misconception
- Object properties: hasFormula, hasSkill, requiresPrerequisiteSkill, hasCommonMisconception
- Data properties: formulaText, description, skillID, difficultyLevel
- Shape individuals: CircleConcept, RectangleConcept, TriangleConcept, ParalleolgramConcept (as in your file)
"""
