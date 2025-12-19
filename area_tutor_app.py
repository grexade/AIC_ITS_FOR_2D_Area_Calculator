

from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL


# -----------------------------
# Model layer
# -----------------------------

@dataclass
class Concept:
    """A learnable concept loaded from the ontology (e.g., RectangleConcept)."""
    name: str
    uri: URIRef
    shape_type: str  # "Rectangle" | "Triangle" | "Circle" | "Parallelogram" | "CompositeShape"
    formula_text: str = ""
    difficulty: Optional[int] = None


@dataclass
class Problem:
    """Generated problem instance for a concept."""
    concept: Concept
    prompt: str
    correct_answer: float
    dims: Dict[str, float]  # e.g., {"length":9, "width":6} or {"radius":7}
    created_at: float = field(default_factory=time.time)


@dataclass
class StudentModel:
    """Tracks learner performance across concepts."""
    level: str = "Novice"  # Novice / Intermediate / Advanced
    mastery: Dict[str, Dict[str, float]] = field(default_factory=dict)
    misconceptions: Dict[str, int] = field(default_factory=dict)

    def ensure_concept(self, concept_name: str) -> None:
        if concept_name not in self.mastery:
            self.mastery[concept_name] = {"attempts": 0, "correct": 0, "avg_time": 0.0}

    def log_attempt(self, concept_name: str, correct: bool, elapsed: float) -> None:
        self.ensure_concept(concept_name)
        m = self.mastery[concept_name]
        m["attempts"] += 1
        if correct:
            m["correct"] += 1
        # Running average time
        n = m["attempts"]
        m["avg_time"] = ((m["avg_time"] * (n - 1)) + elapsed) / n

    def log_misconception(self, code: str) -> None:
        self.misconceptions[code] = self.misconceptions.get(code, 0) + 1


class OntologyManager:
    """
    Loads and queries the OWL ontology using rdflib.

    It discovers "concepts" as individuals typed by subclasses of Shape
    (e.g., CircleConcept typed Circle, RectangleConcept typed Rectangle, etc.).
    """
    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path
        self.g = Graph()
        self.g.parse(ontology_path)

        # Your ontology namespace
        self.A = Namespace("http://www.semanticweb.org/user/ontologies/2025/10/area2d_ontology.owl#")

    def _localname(self, uri: URIRef) -> str:
        s = str(uri)
        return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

    def get_concepts(self) -> List[Concept]:
        """
        Returns concepts as shape individuals (e.g., RectangleConcept).
        Implementation:
        1) Find all classes that are direct subclasses of Shape
        2) Find individuals typed by those classes
        3) Extract their formulaText via hasFormula
        """
        A = self.A
        g = self.g

        # Subclasses of Shape
        shape_classes = set(g.subjects(RDFS.subClassOf, A.Shape))

        concepts: List[Concept] = []

        for cls in shape_classes:
            shape_type = self._localname(cls)

            for ind in g.subjects(RDF.type, cls):
                # Get formulaText (if linked)
                formula_text = ""
                for f in g.objects(ind, A.hasFormula):
                    ft = g.value(f, A.formulaText)
                    if ft is not None:
                        formula_text = str(ft)
                        break

                # Difficulty (if there is a skill linked)
                difficulty = None
                for sk in g.objects(ind, A.hasSkill):
                    dl = g.value(sk, A.difficultyLevel)
                    if dl is not None:
                        try:
                            difficulty = int(dl)
                        except Exception:
                            difficulty = None
                        break

                concepts.append(
                    Concept(
                        name=self._localname(ind),
                        uri=ind,
                        shape_type=shape_type,
                        formula_text=formula_text,
                        difficulty=difficulty
                    )
                )

        concepts.sort(key=lambda c: c.name.lower())
        return concepts


class TutorEngine:
    """
    Generates problems and diagnoses answers (rule-based).
    Keeps the math in code while pulling formulas from ontology for explainable hints.
    """
    PI_APPROX = 3.14159

    def __init__(self, student: StudentModel):
        self.student = student

    def generate_problem(self, concept: Concept) -> Problem:
        level = self.student.level

        # Difficulty scaling (simple)
        if level == "Novice":
            lo, hi = 3, 9
        elif level == "Intermediate":
            lo, hi = 5, 12
        else:
            lo, hi = 8, 18

        shape = concept.shape_type.lower()

        if shape == "rectangle":
            length = random.randint(lo, hi)
            width = random.randint(lo, hi)
            ans = float(length * width)
            prompt = f"Find the area of a rectangle with Length={length} and Width={width}."
            dims = {"length": float(length), "width": float(width)}
            return Problem(concept, prompt, ans, dims)

        if shape == "triangle":
            base = random.randint(lo, hi)
            height = random.randint(lo, hi)
            ans = float(0.5 * base * height)
            prompt = f"Find the area of a triangle with Base={base} and Height={height}."
            dims = {"base": float(base), "height": float(height)}
            return Problem(concept, prompt, ans, dims)

        if shape == "circle":
            r = random.randint(lo, hi)
            ans = float(self.PI_APPROX * (r ** 2))
            prompt = f"Find the area of a circle with Radius={r}."
            dims = {"radius": float(r)}
            return Problem(concept, prompt, ans, dims)

        if shape == "parallelogram":
            base = random.randint(lo, hi)
            height = random.randint(lo, hi)
            ans = float(base * height)
            prompt = f"Find the area of a parallelogram with Base={base} and Height={height}."
            dims = {"base": float(base), "height": float(height)}
            return Problem(concept, prompt, ans, dims)

        # Fallback
        prompt = "Concept not supported yet."
        return Problem(concept, prompt, 0.0, {})

    def diagnose(self, problem: Problem, user_value: float) -> Tuple[bool, str, str]:
        """
        Returns (correct, code, message)
        - code is a misconception tag for logging (or 'OK', 'GENERIC')
        """
        c = problem.concept.shape_type.lower()
        correct = problem.correct_answer

        # tolerance for float answers
        tol = 0.05 * max(1.0, abs(correct))

        if abs(user_value - correct) <= tol:
            return True, "OK", "✅ Correct! Well done."

        # Specific misconception rules
        if c == "triangle":
            b = problem.dims["base"]
            h = problem.dims["height"]
            missing_half = b * h
            if abs(user_value - missing_half) <= tol:
                return False, "TRI_MISSING_HALF", (
                    "You used Base × Height, which is the rectangle-style product.\n"
                    "For triangles, remember the ½ factor: Area = ½ × base × height."
                )

        if c == "rectangle":
            L = problem.dims["length"]
            W = problem.dims["width"]
            perimeter = 2 * (L + W)
            if abs(user_value - perimeter) <= tol:
                return False, "RECT_PERIMETER", (
                    "It looks like you calculated the perimeter (distance around).\n"
                    "For area, multiply: Area = Length × Width."
                )

        if c == "circle":
            r = problem.dims["radius"]
            # common mistake: using diameter instead of radius
            diameter = 2 * r
            wrong = self.PI_APPROX * (diameter ** 2)
            if abs(user_value - wrong) <= tol:
                return False, "CIRC_DIAMETER", (
                    "It looks like you used the diameter instead of the radius.\n"
                    "Circle area uses the radius: Area = π × r²."
                )

        if c == "parallelogram":
            # common mistake: adding base + height (not typical but plausible)
            b = problem.dims["base"]
            h = problem.dims["height"]
            if abs(user_value - (b + h)) <= tol:
                return False, "PARA_ADD", (
                    "You added base and height, but area needs multiplication.\n"
                    "Area = Base × Height."
                )

        return False, "GENERIC", "❌ Not quite. Re-check the formula and substitute the values carefully."


# -----------------------------
# View + Controller layer (Tkinter)
# -----------------------------

class Style:
    """Centralised style tokens for a clean white + teal look."""
    TEAL = "#005c5c"
    TEAL_LIGHT = "#d6f2f2"
    BG = "#f5f8fa"
    BORDER = "#dee4ea"
    TEXT = "#222831"
    MUTED = "#636b74"

    @staticmethod
    def apply(root: tk.Tk) -> None:
        root.configure(bg=Style.BG)
        s = ttk.Style()

        # Use a modern theme if available
        try:
            s.theme_use("clam")
        except Exception:
            pass

        s.configure("TFrame", background=Style.BG)
        s.configure("Card.TFrame", background="white", borderwidth=1, relief="solid")
        s.configure("TLabel", background=Style.BG, foreground=Style.TEXT, font=("Arial", 11))
        s.configure("Muted.TLabel", background=Style.BG, foreground=Style.MUTED, font=("Arial", 10))
        s.configure("H1.TLabel", background=Style.BG, foreground=Style.TEXT, font=("Arial", 16, "bold"))
        s.configure("H2.TLabel", background="white", foreground=Style.TEXT, font=("Arial", 12, "bold"))
        s.configure("TButton", font=("Arial", 11))
        s.configure("Primary.TButton", background=Style.TEAL, foreground="white")
        s.map("Primary.TButton", background=[("active", Style.TEAL)])


class App(tk.Tk):
    def __init__(self, ontology_path: str):
        super().__init__()
        self.title("2D Shapes Area Tutor")
        self.geometry("1100x720")
        self.minsize(1000, 680)

        Style.apply(self)

        # Models
        self.onto = OntologyManager(ontology_path)
        self.student = StudentModel(level="Novice")
        self.tutor = TutorEngine(self.student)

        # Controller state
        self.concepts: List[Concept] = self.onto.get_concepts()
        self.current_concept: Optional[Concept] = None
        self.current_problem: Optional[Problem] = None

        # Layout
        self._build_navbar()
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Pages
        self.pages: Dict[str, ttk.Frame] = {}
        self.pages["menu"] = ConceptMenuPage(self.container, self)
        self.pages["practice"] = PracticePage(self.container, self)

        for p in self.pages.values():
            p.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Start with Dip-Test (modal), then menu
        self.after(200, self.show_diptest)

    def _build_navbar(self) -> None:
        bar = tk.Frame(self, bg=Style.TEAL, height=54)
        bar.pack(fill="x", side="top")

        title = tk.Label(bar, text="2D Shapes Area Tutor", bg=Style.TEAL, fg="white",
                         font=("Arial", 16, "bold"))
        title.pack(side="left", padx=16, pady=10)

        self.level_label = tk.Label(bar, text="Level: Novice", bg=Style.TEAL, fg="white",
                                    font=("Arial", 11))
        self.level_label.pack(side="right", padx=16)

    def set_level(self, level: str) -> None:
        self.student.level = level
        self.level_label.configure(text=f"Level: {level}")

    def show_diptest(self) -> None:
        DipTestModal(self, on_done=self._diptest_done)

    def _diptest_done(self, level: str) -> None:
        self.set_level(level)
        self.show_page("menu")

    def show_page(self, key: str) -> None:
        self.pages[key].tkraise()

    def start_concept(self, concept: Concept) -> None:
        self.current_concept = concept
        self.current_problem = self.tutor.generate_problem(concept)
        practice: PracticePage = self.pages["practice"]  # type: ignore
        practice.load_problem(self.current_problem)
        self.show_page("practice")

    def next_problem(self) -> None:
        if not self.current_concept:
            return
        self.current_problem = self.tutor.generate_problem(self.current_concept)
        practice: PracticePage = self.pages["practice"]  # type: ignore
        practice.load_problem(self.current_problem)


class DipTestModal(tk.Toplevel):
    """Simple 3-question diagnostic to initialise student level."""
    def __init__(self, parent: App, on_done):
        super().__init__(parent)
        self.title("Diagnostic Dip-Test")
        self.resizable(False, False)
        self.configure(bg=Style.BG)
        self.on_done = on_done

        self.transient(parent)
        self.grab_set()

        w, h = 640, 420
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        ttk.Label(self, text="Diagnostic Assessment (Dip-Test)", style="H1.TLabel").pack(anchor="w", padx=18, pady=(16, 6))
        ttk.Label(self, text="Answer 3 quick questions to set your starting level.", style="Muted.TLabel").pack(anchor="w", padx=18)

        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=18, pady=16)

        self.score = 0
        self.vars: List[tk.IntVar] = [tk.IntVar(value=-1) for _ in range(3)]

        questions = [
            ("Area of a rectangle is:", ["Length × Width", "2(L+W)", "L ÷ W"], 0),
            ("Triangle area includes:", ["½ factor", "2× factor", "No factor"], 0),
            ("Circle area needs:", ["Radius", "Diameter", "Perimeter"], 0),
        ]

        for i, (q, opts, correct) in enumerate(questions):
            row = ttk.Frame(card)
            row.pack(fill="x", padx=14, pady=(10 if i == 0 else 8, 0))
            ttk.Label(row, text=f"Q{i+1}. {q}", style="H2.TLabel").pack(anchor="w")

            opt_row = ttk.Frame(card)
            opt_row.pack(fill="x", padx=14, pady=6)

            for j, opt in enumerate(opts):
                rb = ttk.Radiobutton(opt_row, text=opt, variable=self.vars[i], value=j)
                rb.pack(side="left", padx=(0, 18))

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=18, pady=(0, 16))

        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="right", padx=(8, 0))
        ttk.Button(actions, text="Submit", style="Primary.TButton", command=self._submit).pack(side="right")

    def _cancel(self) -> None:
        # default placement if they cancel
        self.on_done("Novice")
        self.destroy()

    def _submit(self) -> None:
        answers = [v.get() for v in self.vars]
        if any(a == -1 for a in answers):
            messagebox.showwarning("Incomplete", "Please answer all questions.")
            return

        # scoring (correct answers are option 0 for these Qs)
        score = sum(1 for a in answers if a == 0)

        if score <= 1:
            level = "Novice"
        elif score == 2:
            level = "Intermediate"
        else:
            level = "Advanced"

        self.on_done(level)
        self.destroy()


class ConceptMenuPage(ttk.Frame):
    """Ontology-driven concept menu (dynamic, not hard-coded)."""
    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app

        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(18, 8))
        ttk.Label(header, text="Choose a Shape", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Concepts are loaded dynamically from the ontology (subclasses of Shape).",
            style="Muted.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=18, pady=10)

        left = ttk.Frame(body, style="Card.TFrame")
        left.pack(side="left", fill="y", padx=(0, 12))
        ttk.Label(left, text="Concepts", style="H2.TLabel").pack(anchor="w", padx=14, pady=(12, 2))
        ttk.Label(left, text="Ontology → Shape individuals", style="Muted.TLabel").pack(anchor="w", padx=14, pady=(0, 10))

        self.list_frame = ttk.Frame(left)
        self.list_frame.pack(fill="y", padx=14, pady=(0, 14))

        right = ttk.Frame(body, style="Card.TFrame")
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="How this is ontology-driven", style="H2.TLabel").pack(anchor="w", padx=14, pady=(12, 4))
        ttk.Label(
            right,
            text=(
                "The tutor reads the OWL file at runtime.\n"
                "If you add a new shape under Shape in Protégé,\n"
                "it will appear here automatically without changing Python code."
            ),
        ).pack(anchor="w", padx=14, pady=(0, 12))

        self._populate()

    def _populate(self) -> None:
        # clear
        for w in self.list_frame.winfo_children():
            w.destroy()

        if not self.app.concepts:
            ttk.Label(self.list_frame, text="No concepts found in ontology.").pack(anchor="w")
            return

        for c in self.app.concepts:
            row = ttk.Frame(self.list_frame)
            row.pack(fill="x", pady=6)

            label = ttk.Label(row, text=f"{c.shape_type}  •  {c.name}")
            label.pack(side="left")

            ttk.Button(row, text="Open", style="Primary.TButton", command=lambda cc=c: self.app.start_concept(cc)).pack(side="right")


class PracticePage(ttk.Frame):
    """Main tutoring interface: canvas + input + hint + feedback."""
    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app

        top = ttk.Frame(self)
        top.pack(fill="x", padx=18, pady=(18, 6))

        self.title = ttk.Label(top, text="Practice", style="H1.TLabel")
        self.title.pack(side="left")

        ttk.Button(top, text="Back to menu", command=lambda: self.app.show_page("menu")).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=18, pady=10)

        # Left card: canvas
        self.left = ttk.Frame(body, style="Card.TFrame")
        self.left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        ttk.Label(self.left, text="Visual Model", style="H2.TLabel").pack(anchor="w", padx=14, pady=(12, 6))
        self.canvas = tk.Canvas(self.left, bg="#f8fafc", highlightthickness=1, highlightbackground=Style.BORDER, height=320)
        self.canvas.pack(fill="x", padx=14, pady=(0, 10))

        ttk.Label(self.left, text="Scaffolding steps", style="H2.TLabel").pack(anchor="w", padx=14, pady=(6, 6))
        self.steps = tk.Text(self.left, height=6, wrap="word")
        self.steps.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.steps.configure(state="disabled")

        # Right card: input + feedback
        self.right = ttk.Frame(body, style="Card.TFrame")
        self.right.pack(side="left", fill="both", expand=True)

        ttk.Label(self.right, text="Problem", style="H2.TLabel").pack(anchor="w", padx=14, pady=(12, 6))
        self.problem_lbl = ttk.Label(self.right, text="", wraplength=430)
        self.problem_lbl.pack(anchor="w", padx=14)

        self.hint_box = ttk.Label(self.right, text="", background=Style.TEAL_LIGHT, foreground=Style.TEAL,
                                  wraplength=430, padding=10)
        self.hint_box.pack(fill="x", padx=14, pady=(12, 12))

        ttk.Label(self.right, text="Your answer (square units)", style="H2.TLabel").pack(anchor="w", padx=14)
        self.answer_var = tk.StringVar(value="")
        self.answer_entry = ttk.Entry(self.right, textvariable=self.answer_var, width=18, font=("Arial", 12))
        self.answer_entry.pack(anchor="w", padx=14, pady=8)

        actions = ttk.Frame(self.right)
        actions.pack(fill="x", padx=14, pady=6)

        ttk.Button(actions, text="Submit", style="Primary.TButton", command=self.on_submit).pack(side="left")
        ttk.Button(actions, text="New problem", command=self.app.next_problem).pack(side="left", padx=8)

        self.feedback = ttk.Label(self.right, text="", wraplength=430)
        self.feedback.pack(anchor="w", padx=14, pady=(12, 14))

    def load_problem(self, problem: Problem) -> None:
        self.app.current_problem = problem
        c = problem.concept

        self.title.configure(text=f"Practice • {c.shape_type}")
        self.problem_lbl.configure(text=problem.prompt)

        # Ontology hint (formulaText)
        hint = c.formula_text.strip() if c.formula_text else f"Formula: {self._fallback_formula(c.shape_type)}"
        self.hint_box.configure(text=f"Ontology hint (formulaText): {hint}")

        self.answer_var.set("")
        self.feedback.configure(text="")

        self._draw(problem)
        self._set_steps(problem)

        self.answer_entry.focus_set()

    def _fallback_formula(self, shape_type: str) -> str:
        st = shape_type.lower()
        if st == "rectangle":
            return "Area = Length × Width"
        if st == "triangle":
            return "Area = ½ × Base × Height"
        if st == "circle":
            return "Area = π × r²"
        if st == "parallelogram":
            return "Area = Base × Height"
        return "Area formula"

    def _set_steps(self, problem: Problem) -> None:
        c = problem.concept.shape_type.lower()
        if c == "rectangle":
            L, W = problem.dims["length"], problem.dims["width"]
            lines = [
                "1) Select the rectangle area formula: Area = Length × Width",
                f"2) Substitute values: {L:g} × {W:g}",
                f"3) Compute: {L*g(W):.0f} square units" if False else "3) Compute the product to get the area."
            ]
        elif c == "triangle":
            b, h = problem.dims["base"], problem.dims["height"]
            lines = [
                "1) Select the triangle area formula: Area = ½ × base × height",
                f"2) Substitute values: ½ × {b:g} × {h:g}",
                "3) Multiply base × height, then divide by 2."
            ]
        elif c == "circle":
            r = problem.dims["radius"]
            lines = [
                "1) Select the circle area formula: Area = π × r²",
                f"2) Substitute values: 3.14159 × {r:g}²",
                "3) Square the radius, then multiply by π."
            ]
        else:
            lines = [
                "1) Identify the correct area formula.",
                "2) Substitute the values.",
                "3) Compute the final answer."
            ]

        self.steps.configure(state="normal")
        self.steps.delete("1.0", tk.END)
        self.steps.insert("1.0", "\n".join(lines))
        self.steps.configure(state="disabled")

    def _draw(self, problem: Problem) -> None:
        self.canvas.delete("all")

        shape = problem.concept.shape_type.lower()
        w = int(self.canvas.winfo_width() or 700)
        h = int(self.canvas.winfo_height() or 320)

        pad = 30
        cx, cy = w // 2, h // 2

        if shape == "rectangle":
            L = problem.dims["length"]
            Wd = problem.dims["width"]

            # scale
            max_dim = max(L, Wd)
            scale = (min(w, h) - 2 * pad) / max_dim

            rw = L * scale
            rh = Wd * scale
            x1, y1 = cx - rw / 2, cy - rh / 2
            x2, y2 = cx + rw / 2, cy + rh / 2

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=Style.TEAL, width=4)
            self.canvas.create_text(cx, y2 + 16, text=f"Length={L:g}", fill=Style.TEXT, font=("Arial", 11))
            self.canvas.create_text(x2 + 55, cy, text=f"Width={Wd:g}", fill=Style.TEXT, font=("Arial", 11))

        elif shape == "triangle":
            b = problem.dims["base"]
            ht = problem.dims["height"]

            scale = (min(w, h) - 2 * pad) / max(b, ht, 1.0)

            base_px = b * scale
            height_px = ht * scale

            x1, y1 = cx - base_px / 2, cy + height_px / 2
            x2, y2 = cx + base_px / 2, cy + height_px / 2
            x3, y3 = cx, cy - height_px / 2

            self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, outline=Style.TEAL, fill="", width=4)
            self.canvas.create_text(cx, y2 + 16, text=f"Base={b:g}", fill=Style.TEXT, font=("Arial", 11))
            self.canvas.create_text(cx + 80, cy, text=f"Height={ht:g}", fill=Style.TEXT, font=("Arial", 11))

        elif shape == "circle":
            r = problem.dims["radius"]
            scale = (min(w, h) - 2 * pad) / (2 * r)
            rp = r * scale

            self.canvas.create_oval(cx - rp, cy - rp, cx + rp, cy + rp, outline=Style.TEAL, width=4)
            self.canvas.create_line(cx, cy, cx + rp, cy, fill=Style.TEAL, width=3)
            self.canvas.create_text(cx + rp / 2, cy - 14, text=f"r={r:g}", fill=Style.TEXT, font=("Arial", 11))

        elif shape == "parallelogram":
            b = problem.dims["base"]
            ht = problem.dims["height"]
            scale = (min(w, h) - 2 * pad) / max(b, ht, 1.0)

            base_px = b * scale
            height_px = ht * scale
            slant = base_px * 0.25

            x1, y1 = cx - base_px / 2, cy + height_px / 2
            x2, y2 = cx + base_px / 2, cy + height_px / 2
            x3, y3 = x2 + slant, cy - height_px / 2
            x4, y4 = x1 + slant, cy - height_px / 2

            self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, x4, y4, outline=Style.TEAL, fill="", width=4)
            self.canvas.create_text(cx, y2 + 16, text=f"Base={b:g}", fill=Style.TEXT, font=("Arial", 11))
            self.canvas.create_text(cx + 95, cy, text=f"Height={ht:g}", fill=Style.TEXT, font=("Arial", 11))

        else:
            self.canvas.create_text(cx, cy, text="No visual available for this concept.", fill=Style.MUTED)

    def on_submit(self) -> None:
        problem = self.app.current_problem
        if not problem:
            return

        raw = self.answer_var.get().strip()

        # Input validation
        try:
            val = float(raw)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value (e.g., 54 or 153.94).")
            return

        # Diagnose
        correct, code, msg = self.app.tutor.diagnose(problem, val)

        # Log
        elapsed = time.time() - problem.created_at
        self.app.student.log_attempt(problem.concept.name, correct, elapsed)
        if code not in ("OK", "GENERIC"):
            self.app.student.log_misconception(code)

        # Feedback + next-step prompt
        self.feedback.configure(text=msg)

        if correct:
            # auto move to next after short delay
            self.after(900, self.app.next_problem)


# -----------------------------
# main
# -----------------------------

def main():
    # Change this to your OWL filename if needed:
    ontology_path = "area2d_ontology.owl"
    app = App(ontology_path=ontology_path)
    app.mainloop()


if __name__ == "__main__":
    main()
