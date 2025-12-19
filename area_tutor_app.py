
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS


# -----------------------------------------------------------------------------
# Model: Ontology objects + Student model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Concept:
    """A learnable concept loaded from the ontology (e.g., RectangleConcept)."""
    name: str                 # local name (e.g., RectangleConcept)
    uri: URIRef               # full URI
    shape_type: str           # Rectangle / Triangle / Circle / Parallelogram / CompositeShape
    formula_text: str = ""    # pulled from ontology via hasFormula/formulaText
    difficulty: Optional[int] = None


@dataclass
class Problem:
    concept: Concept
    prompt: str
    correct_answer: float
    dims: Dict[str, float]
    created_at: float = field(default_factory=time.time)


@dataclass
class StudentModel:
    """Simple learner model for adaptation + reporting."""
    level: str = "Novice"  # Novice / Intermediate / Advanced
    mastery: Dict[str, Dict[str, float]] = field(default_factory=dict)
    misconceptions: Dict[str, int] = field(default_factory=dict)

    def ensure(self, concept_name: str) -> None:
        if concept_name not in self.mastery:
            self.mastery[concept_name] = {"attempts": 0, "correct": 0, "avg_time": 0.0}

    def log_attempt(self, concept_name: str, correct: bool, elapsed: float) -> None:
        self.ensure(concept_name)
        m = self.mastery[concept_name]
        m["attempts"] += 1
        if correct:
            m["correct"] += 1
        n = m["attempts"]
        m["avg_time"] = ((m["avg_time"] * (n - 1)) + elapsed) / n

    def log_misconception(self, code: str) -> None:
        self.misconceptions[code] = self.misconceptions.get(code, 0) + 1

    def stats_text(self, concept_name: str) -> str:
        self.ensure(concept_name)
        m = self.mastery[concept_name]
        attempts = int(m["attempts"])
        correct = int(m["correct"])
        acc = (correct / attempts * 100.0) if attempts else 0.0
        avg_time = m["avg_time"]
        return f"Attempts: {attempts}   •   Accuracy: {acc:.0f}%   •   Avg time: {avg_time:.1f}s"


class OntologyManager:
    """
    Loads and queries the OWL ontology using rdflib.

    Your ontology namespace (from your files):
      http://www.semanticweb.org/user/ontologies/2025/10/area2d_ontology.owl#
    """

    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path
        self.g = Graph()
        self.g.parse(ontology_path)

        self.A = Namespace(
            "http://www.semanticweb.org/user/ontologies/2025/10/area2d_ontology.owl#"
        )

    @staticmethod
    def _local(u: URIRef) -> str:
        s = str(u)
        return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

    def list_concepts(self) -> List[Concept]:
        """
        Discover concepts dynamically:
          - find subclasses of Shape
          - find individuals typed by each subclass
          - attach formulaText (via hasFormula -> formulaText)
          - attach difficulty (via hasSkill -> difficultyLevel) if present
        """
        g, A = self.g, self.A

        shape_subclasses = sorted(
            set(g.subjects(RDFS.subClassOf, A.Shape)),
            key=lambda u: self._local(u).lower(),
        )

        concepts: List[Concept] = []
        for cls in shape_subclasses:
            shape_type = self._local(cls)
            for ind in g.subjects(RDF.type, cls):
                formula_text = self._formula_text_for(ind)
                difficulty = self._difficulty_for(ind)
                concepts.append(
                    Concept(
                        name=self._local(ind),
                        uri=ind,
                        shape_type=shape_type,
                        formula_text=formula_text,
                        difficulty=difficulty,
                    )
                )

        concepts.sort(key=lambda c: (c.shape_type.lower(), c.name.lower()))
        return concepts

    # ---- Info queries for the Ontology Info tab ----

    def shape_info(self, concept_uri: URIRef) -> Dict[str, object]:
        """Return formula(s), skills, prereqs, misconceptions (if present)."""
        g, A = self.g, self.A

        info: Dict[str, object] = {"formulas": [], "skills": []}

        # Formulas
        for f in g.objects(concept_uri, A.hasFormula):
            info["formulas"].append(
                {
                    "id": self._local(f),
                    "formulaText": str(g.value(f, A.formulaText) or ""),
                }
            )

        # Skills
        for sk in g.objects(concept_uri, A.hasSkill):
            prereq = [self._local(x) for x in g.objects(sk, A.requiresPrerequisiteSkill)]
            miscon = [self._local(x) for x in g.objects(sk, A.hasCommonMisconception)]
            info["skills"].append(
                {
                    "id": self._local(sk),
                    "skillID": str(g.value(sk, A.skillID) or ""),
                    "description": str(g.value(sk, A.description) or ""),
                    "difficultyLevel": str(g.value(sk, A.difficultyLevel) or ""),
                    "prerequisites": prereq,
                    "misconceptions": miscon,
                }
            )
        return info

    # ---- Internal helpers ----

    def _formula_text_for(self, ind: URIRef) -> str:
        g, A = self.g, self.A
        for f in g.objects(ind, A.hasFormula):
            ft = g.value(f, A.formulaText)
            if ft:
                return str(ft)
        return ""

    def _difficulty_for(self, ind: URIRef) -> Optional[int]:
        g, A = self.g, self.A
        for sk in g.objects(ind, A.hasSkill):
            dl = g.value(sk, A.difficultyLevel)
            if dl is None:
                continue
            try:
                return int(str(dl))
            except ValueError:
                return None
        return None


# -----------------------------------------------------------------------------
# Tutor engine: problem generation + diagnosis (rule-based)
# -----------------------------------------------------------------------------

class TutorEngine:
    PI = 3.14159

    def __init__(self, student: StudentModel):
        self.student = student

    def _range(self) -> Tuple[int, int]:
        if self.student.level == "Novice":
            return 3, 9
        if self.student.level == "Intermediate":
            return 5, 12
        return 8, 18

    def generate_problem(self, concept: Concept) -> Problem:
        lo, hi = self._range()
        st = concept.shape_type.lower()

        if st == "rectangle":
            L = random.randint(lo, hi)
            W = random.randint(lo, hi)
            ans = float(L * W)
            prompt = f"Find the area of a rectangle with Length={L} and Width={W}."
            return Problem(concept, prompt, ans, {"length": float(L), "width": float(W)})

        if st == "triangle":
            b = random.randint(lo, hi)
            h = random.randint(lo, hi)
            ans = float(0.5 * b * h)
            prompt = f"Find the area of a triangle with Base={b} and Height={h}."
            return Problem(concept, prompt, ans, {"base": float(b), "height": float(h)})

        if st == "circle":
            r = random.randint(lo, hi)
            ans = float(self.PI * (r ** 2))
            prompt = f"Find the area of a circle with Radius={r}."
            return Problem(concept, prompt, ans, {"radius": float(r)})

        if st == "parallelogram":
            b = random.randint(lo, hi)
            h = random.randint(lo, hi)
            ans = float(b * h)
            prompt = f"Find the area of a parallelogram with Base={b} and Height={h}."
            return Problem(concept, prompt, ans, {"base": float(b), "height": float(h)})

        # CompositeShape not implemented in this demo
        prompt = "This concept is not implemented yet."
        return Problem(concept, prompt, 0.0, {})

    def diagnose(self, problem: Problem, user_value: float) -> Tuple[bool, str, str]:
        """
        Returns:
          (is_correct, code, feedback_message)
        code is useful for misconception logging.
        """
        correct = problem.correct_answer
        tol = 0.05 * max(1.0, abs(correct))  # 5% tolerance

        if abs(user_value - correct) <= tol:
            return True, "OK", "✅ Correct! Great job."

        st = problem.concept.shape_type.lower()

        # Misconception checks (buggy rules)
        if st == "triangle":
            b = problem.dims["base"]
            h = problem.dims["height"]
            missing_half = b * h
            if abs(user_value - missing_half) <= tol:
                return (
                    False,
                    "TRI_MISSING_HALF",
                    "You used Base × Height (rectangle-style).\n"
                    "For triangles, remember: Area = ½ × base × height.",
                )

        if st == "rectangle":
            L = problem.dims["length"]
            W = problem.dims["width"]
            perimeter = 2 * (L + W)
            if abs(user_value - perimeter) <= tol:
                return (
                    False,
                    "RECT_PERIMETER",
                    "It looks like you calculated perimeter (distance around).\n"
                    "Area is space inside: Area = Length × Width.",
                )

        if st == "circle":
            r = problem.dims["radius"]
            diameter = 2 * r
            wrong = self.PI * (diameter ** 2)
            if abs(user_value - wrong) <= tol:
                return (
                    False,
                    "CIRC_DIAMETER",
                    "It looks like you used diameter instead of radius.\n"
                    "Circle area uses radius: Area = π × r².",
                )

        if st == "parallelogram":
            b = problem.dims["base"]
            h = problem.dims["height"]
            if abs(user_value - (b + h)) <= tol:
                return (
                    False,
                    "PARA_ADD",
                    "You added base and height. For area, multiply:\n"
                    "Area = Base × Height.",
                )

        # Generic fallback
        return False, "GENERIC", "❌ Not quite. Re-check the formula and your substitution."


# -----------------------------------------------------------------------------
# UI styling: white + teal theme
# -----------------------------------------------------------------------------

class UI:
    TEAL = "#005c5c"
    TEAL_LIGHT = "#d6f2f2"
    BG = "#f5f8fa"
    CARD = "#ffffff"
    BORDER = "#dee4ea"
    TEXT = "#222831"
    MUTED = "#636b74"

    @staticmethod
    def apply_style(root: tk.Tk) -> ttk.Style:
        root.configure(bg=UI.BG)
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=UI.BG)
        style.configure("Card.TFrame", background=UI.CARD, borderwidth=1, relief="solid")
        style.configure("TLabel", background=UI.BG, foreground=UI.TEXT, font=("Arial", 11))
        style.configure("Muted.TLabel", background=UI.BG, foreground=UI.MUTED, font=("Arial", 10))
        style.configure("H1.TLabel", background=UI.BG, foreground=UI.TEXT, font=("Arial", 16, "bold"))
        style.configure("H2.TLabel", background=UI.CARD, foreground=UI.TEXT, font=("Arial", 12, "bold"))
        style.configure("Primary.TButton", background=UI.TEAL, foreground="white", font=("Arial", 11, "bold"))
        style.map("Primary.TButton", background=[("active", UI.TEAL)])
        return style


# -----------------------------------------------------------------------------
# UI components
# -----------------------------------------------------------------------------

class DipTestModal(tk.Toplevel):
    """3-question diagnostic assessment used to set initial student level."""

    def __init__(self, parent: "TutorApp", on_done):
        super().__init__(parent)
        self.configure(bg=UI.BG)
        self.title("Diagnostic Dip-Test")
        self.resizable(False, False)

        self.transient(parent)
        self.grab_set()
        self.on_done = on_done

        # Center modal
        w, h = 640, 420
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        ttk.Label(self, text="Diagnostic Assessment (Dip-Test)", style="H1.TLabel").pack(
            anchor="w", padx=18, pady=(16, 6)
        )
        ttk.Label(
            self,
            text="Answer 3 quick questions to personalise your starting level.",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=18)

        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=18, pady=16)

        self.vars: List[tk.IntVar] = [tk.IntVar(value=-1) for _ in range(3)]

        questions = [
            ("Area of a rectangle is:", ["Length × Width", "2(L+W)", "L ÷ W"], 0),
            ("Triangle area includes:", ["½ factor", "2× factor", "No factor"], 0),
            ("Circle area needs:", ["Radius", "Diameter", "Perimeter"], 0),
        ]

        for i, (q, opts, _correct) in enumerate(questions):
            row = ttk.Frame(card)
            row.pack(fill="x", padx=14, pady=(12 if i == 0 else 10, 0))
            ttk.Label(row, text=f"Q{i+1}. {q}", style="H2.TLabel").pack(anchor="w")

            opt_row = ttk.Frame(card)
            opt_row.pack(fill="x", padx=14, pady=6)
            for j, opt in enumerate(opts):
                ttk.Radiobutton(opt_row, text=opt, variable=self.vars[i], value=j).pack(
                    side="left", padx=(0, 18)
                )

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=18, pady=(0, 16))
        ttk.Button(actions, text="Cancel", command=self._cancel).pack(side="right", padx=(8, 0))
        ttk.Button(actions, text="Submit", style="Primary.TButton", command=self._submit).pack(side="right")

    def _cancel(self):
        self.on_done("Novice")
        self.destroy()

    def _submit(self):
        answers = [v.get() for v in self.vars]
        if any(a == -1 for a in answers):
            messagebox.showwarning("Incomplete", "Please answer all questions.")
            return

        score = sum(1 for a in answers if a == 0)  # correct option is 0 in this dip-test
        level = "Novice" if score <= 1 else ("Intermediate" if score == 2 else "Advanced")
        self.on_done(level)
        self.destroy()


class TutorApp(tk.Tk):
    """Main application window."""

    def __init__(self, ontology_path: str = "area2d_ontology.owl"):
        super().__init__()
        self.title("2D Shapes Area Tutor")
        self.geometry("1120x740")
        self.minsize(1000, 680)

        UI.apply_style(self)

        # Models
        self.onto = OntologyManager(ontology_path)
        self.student = StudentModel(level="Novice")
        self.tutor = TutorEngine(self.student)

        # Domain concepts from ontology
        self.concepts = self.onto.list_concepts()
        if not self.concepts:
            messagebox.showerror(
                "Ontology error",
                "No Shape concepts found. Check your OWL file namespace and individuals.",
            )

        self.current_concept: Optional[Concept] = None
        self.current_problem: Optional[Problem] = None

        # Header
        self._build_header()

        # Tabs (Practice + Ontology Info)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=16, pady=(10, 16))

        self.practice_tab = PracticeTab(self.notebook, self)
        self.info_tab = OntologyInfoTab(self.notebook, self)

        self.notebook.add(self.practice_tab, text="Practice")
        self.notebook.add(self.info_tab, text="Ontology Info")

        # Start with dip-test
        self.after(200, self._show_diptest)

    def _build_header(self):
        bar = tk.Frame(self, bg=UI.TEAL, height=56)
        bar.pack(fill="x", side="top")

        tk.Label(
            bar,
            text="2D Shapes Area Tutor",
            bg=UI.TEAL,
            fg="white",
            font=("Arial", 16, "bold"),
        ).pack(side="left", padx=16, pady=12)

        self.level_label = tk.Label(
            bar, text="Level: Novice", bg=UI.TEAL, fg="white", font=("Arial", 11)
        )
        self.level_label.pack(side="right", padx=16)

    def _show_diptest(self):
        DipTestModal(self, on_done=self._diptest_done)

    def _diptest_done(self, level: str):
        self.student.level = level
        self.level_label.configure(text=f"Level: {level}")
        # After dip-test, focus practice tab
        self.notebook.select(self.practice_tab)

    def start_concept(self, concept: Concept):
        self.current_concept = concept
        self.next_problem()

    def next_problem(self):
        if not self.current_concept:
            return
        self.current_problem = self.tutor.generate_problem(self.current_concept)
        self.practice_tab.load_problem(self.current_problem)


class PracticeTab(ttk.Frame):
    """Practice workflow: concept list + tutoring panel."""

    def __init__(self, parent, app: TutorApp):
        super().__init__(parent)
        self.app = app

        header = ttk.Frame(self)
        header.pack(fill="x", padx=14, pady=(14, 6))
        ttk.Label(header, text="Choose a concept and practise area problems", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Concepts are loaded dynamically from the ontology (subclasses of Shape).",
            style="Muted.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        # Left: Concept list
        self.left = ttk.Frame(body, style="Card.TFrame")
        self.left.pack(side="left", fill="y", padx=(0, 12))
        ttk.Label(self.left, text="Concepts", style="H2.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
        ttk.Label(self.left, text="Ontology → Shape individuals", style="Muted.TLabel").pack(anchor="w", padx=12, pady=(0, 10))

        self.concept_list = tk.Listbox(self.left, height=18, font=("Arial", 11))
        self.concept_list.pack(fill="y", padx=12, pady=(0, 12))
        self._populate_concepts()

        ttk.Button(self.left, text="Start", style="Primary.TButton", command=self._start_selected).pack(
            fill="x", padx=12, pady=(0, 12)
        )

        # Right: Tutoring panel
        self.right = ttk.Frame(body, style="Card.TFrame")
        self.right.pack(side="left", fill="both", expand=True)

        ttk.Label(self.right, text="Problem", style="H2.TLabel").pack(anchor="w", padx=12, pady=(12, 6))
        self.problem_lbl = ttk.Label(self.right, text="", wraplength=560)
        self.problem_lbl.pack(anchor="w", padx=12)

        self.hint_lbl = ttk.Label(
            self.right,
            text="Ontology hint will appear here.",
            background=UI.TEAL_LIGHT,
            foreground=UI.TEAL,
            padding=10,
            wraplength=560,
        )
        self.hint_lbl.pack(fill="x", padx=12, pady=(10, 10))

        # Canvas visual
        ttk.Label(self.right, text="Visual Model", style="H2.TLabel").pack(anchor="w", padx=12, pady=(2, 6))
        self.canvas = tk.Canvas(self.right, bg="#f8fafc", height=260, highlightthickness=1, highlightbackground=UI.BORDER)
        self.canvas.pack(fill="x", padx=12, pady=(0, 10))

        # Answer
        ttk.Label(self.right, text="Your answer (square units)", style="H2.TLabel").pack(anchor="w", padx=12, pady=(0, 2))
        row = ttk.Frame(self.right)
        row.pack(fill="x", padx=12, pady=(0, 10))

        self.answer_var = tk.StringVar(value="")
        self.answer_entry = ttk.Entry(row, textvariable=self.answer_var, width=18, font=("Arial", 12))
        self.answer_entry.pack(side="left")

        ttk.Button(row, text="Submit", style="Primary.TButton", command=self._submit).pack(side="left", padx=8)
        ttk.Button(row, text="New problem", command=self._new_problem).pack(side="left")

        # Feedback + progress
        self.feedback_lbl = ttk.Label(self.right, text="", wraplength=560)
        self.feedback_lbl.pack(anchor="w", padx=12, pady=(2, 6))

        self.progress_lbl = ttk.Label(self.right, text="", style="Muted.TLabel")
        self.progress_lbl.pack(anchor="w", padx=12, pady=(0, 12))

    def _populate_concepts(self):
        self.concept_list.delete(0, tk.END)
        for c in self.app.concepts:
            self.concept_list.insert(tk.END, f"{c.shape_type} • {c.name}")

    def _start_selected(self):
        idxs = self.concept_list.curselection()
        if not idxs:
            messagebox.showinfo("Select a concept", "Please select a concept first.")
            return
        idx = idxs[0]
        concept = self.app.concepts[idx]
        self.app.start_concept(concept)

    def _new_problem(self):
        if not self.app.current_concept:
            messagebox.showinfo("Start first", "Choose a concept and click Start.")
            return
        self.app.next_problem()

    def load_problem(self, problem: Problem):
        self.problem_lbl.configure(text=problem.prompt)

        hint = problem.concept.formula_text.strip() or self._fallback_formula(problem.concept.shape_type)
        self.hint_lbl.configure(text=f"Ontology hint (formulaText): {hint}")

        self.answer_var.set("")
        self.feedback_lbl.configure(text="")
        self.progress_lbl.configure(text=self.app.student.stats_text(problem.concept.name))

        self._draw(problem)
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

    def _draw(self, problem: Problem):
        self.canvas.delete("all")
        st = problem.concept.shape_type.lower()

        # canvas bounds
        w = int(self.canvas.winfo_width() or 720)
        h = int(self.canvas.winfo_height() or 260)
        pad = 22
        cx, cy = w // 2, h // 2

        if st == "rectangle":
            L = problem.dims["length"]
            Wd = problem.dims["width"]
            scale = (min(w, h) - 2 * pad) / max(L, Wd, 1.0)
            rw, rh = L * scale, Wd * scale
            x1, y1 = cx - rw / 2, cy - rh / 2
            x2, y2 = cx + rw / 2, cy + rh / 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=UI.TEAL, width=4)
            self.canvas.create_text(cx, y2 + 14, text=f"Length={L:g}", fill=UI.TEXT, font=("Arial", 11))
            self.canvas.create_text(x2 + 58, cy, text=f"Width={Wd:g}", fill=UI.TEXT, font=("Arial", 11))

        elif st == "triangle":
            b = problem.dims["base"]
            ht = problem.dims["height"]
            scale = (min(w, h) - 2 * pad) / max(b, ht, 1.0)
            base_px, height_px = b * scale, ht * scale
            x1, y1 = cx - base_px / 2, cy + height_px / 2
            x2, y2 = cx + base_px / 2, cy + height_px / 2
            x3, y3 = cx, cy - height_px / 2
            self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, outline=UI.TEAL, fill="", width=4)
            self.canvas.create_text(cx, y2 + 14, text=f"Base={b:g}", fill=UI.TEXT, font=("Arial", 11))
            self.canvas.create_text(cx + 86, cy, text=f"Height={ht:g}", fill=UI.TEXT, font=("Arial", 11))

        elif st == "circle":
            r = problem.dims["radius"]
            scale = (min(w, h) - 2 * pad) / (2 * r if r else 1.0)
            rp = r * scale
            self.canvas.create_oval(cx - rp, cy - rp, cx + rp, cy + rp, outline=UI.TEAL, width=4)
            self.canvas.create_line(cx, cy, cx + rp, cy, fill=UI.TEAL, width=3)
            self.canvas.create_text(cx + rp / 2, cy - 14, text=f"r={r:g}", fill=UI.TEXT, font=("Arial", 11))

        elif st == "parallelogram":
            b = problem.dims["base"]
            ht = problem.dims["height"]
            scale = (min(w, h) - 2 * pad) / max(b, ht, 1.0)
            base_px, height_px = b * scale, ht * scale
            slant = base_px * 0.25
            x1, y1 = cx - base_px / 2, cy + height_px / 2
            x2, y2 = cx + base_px / 2, cy + height_px / 2
            x3, y3 = x2 + slant, cy - height_px / 2
            x4, y4 = x1 + slant, cy - height_px / 2
            self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, x4, y4, outline=UI.TEAL, fill="", width=4)
            self.canvas.create_text(cx, y2 + 14, text=f"Base={b:g}", fill=UI.TEXT, font=("Arial", 11))
            self.canvas.create_text(cx + 98, cy, text=f"Height={ht:g}", fill=UI.TEXT, font=("Arial", 11))

        else:
            self.canvas.create_text(cx, cy, text="No visual available for this concept.", fill=UI.MUTED)

    def _submit(self):
        problem = self.app.current_problem
        if not problem:
            messagebox.showinfo("Start first", "Choose a concept and click Start.")
            return

        raw = self.answer_var.get().strip()
        try:
            val = float(raw)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a numeric value (e.g., 54 or 153.94).")
            return

        correct, code, msg = self.app.tutor.diagnose(problem, val)

        # log
        elapsed = time.time() - problem.created_at
        self.app.student.log_attempt(problem.concept.name, correct, elapsed)
        if code not in ("OK", "GENERIC"):
            self.app.student.log_misconception(code)

        self.feedback_lbl.configure(text=msg)
        self.progress_lbl.configure(text=self.app.student.stats_text(problem.concept.name))

        if correct:
            # next problem after a short delay
            self.after(900, self._new_problem)


class OntologyInfoTab(ttk.Frame):
    """A clean ontology browser tab (useful as evidence for integration in your dissertation)."""

    def __init__(self, parent, app: TutorApp):
        super().__init__(parent)
        self.app = app

        header = ttk.Frame(self)
        header.pack(fill="x", padx=14, pady=(14, 6))
        ttk.Label(header, text="Ontology Browser", style="H1.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Pick a concept to view its formulas and linked skills (from the OWL file).",
            style="Muted.TLabel",
        ).pack(anchor="w")

        card = ttk.Frame(self, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=14, pady=10)

        row = ttk.Frame(card)
        row.pack(fill="x", padx=12, pady=(12, 6))

        ttk.Label(row, text="Select concept:", style="H2.TLabel").pack(side="left")
        self.combo = ttk.Combobox(row, state="readonly", width=40)
        self.combo.pack(side="left", padx=8)

        ttk.Button(row, text="Show Info", style="Primary.TButton", command=self._show).pack(side="left", padx=8)

        self.text = tk.Text(card, wrap="word", font=("Arial", 11))
        self.text.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        # populate dropdown
        self.concepts = self.app.concepts
        self.combo["values"] = [f"{c.shape_type} • {c.name}" for c in self.concepts]
        if self.concepts:
            self.combo.current(0)
            self._show()

    def _show(self):
        if not self.concepts:
            return
        idx = self.combo.current()
        concept = self.concepts[idx]
        info = self.app.onto.shape_info(concept.uri)

        lines: List[str] = []
        lines.append(f"Concept: {concept.name}   (Type: {concept.shape_type})")
        lines.append("-" * 72)
        lines.append("")
        lines.append("Formulas:")
        formulas = info.get("formulas", [])
        if formulas:
            for f in formulas:
                lines.append(f"  • {f.get('formulaText','').strip()}")
        else:
            lines.append("  (No formulas linked via hasFormula)")
        lines.append("")
        lines.append("Skills:")
        skills = info.get("skills", [])
        if skills:
            for s in skills:
                sid = s.get("skillID") or s.get("id") or ""
                desc = s.get("description") or ""
                dl = s.get("difficultyLevel") or ""
                prereq = s.get("prerequisites") or []
                miscon = s.get("misconceptions") or []
                lines.append(f"  Skill: {sid}")
                if desc:
                    lines.append(f"    Description: {desc}")
                if dl:
                    lines.append(f"    Difficulty: {dl}")
                lines.append(f"    Prerequisites: {', '.join(prereq) if prereq else '(none)'}")
                lines.append(f"    Misconceptions: {', '.join(miscon) if miscon else '(none)'}")
                lines.append("")
        else:
            lines.append("  (No skills linked via hasSkill)")
        lines.append("")
        lines.append("Note: If you add prerequisites or misconceptions in Protégé, this tab will display them automatically.")

        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", "\n".join(lines))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    
    ontology_path = "area2d_ontology.owl"
    app = TutorApp(ontology_path=ontology_path)
    app.mainloop()


if __name__ == "__main__":
    main()
