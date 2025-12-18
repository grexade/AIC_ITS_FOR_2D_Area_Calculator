"""
Simple Ontology Browser for Area of 2D Shapes
---------------------------------------------
- Loads an OWL ontology built in Protégé (area2d_ontology.owl)
- Lets the user pick a Shape individual
- Shows formula, skills, prerequisites, and misconceptions
"""

import tkinter as tk
from tkinter import ttk, messagebox

import rdflib as rf
from rdflib import Graph, Namespace


class AreaOntologyInterface:
    """
    Handles loading and querying the OWL ontology.
    This is your 'Domain Model' connection in code.
    """

    def __init__(self, ontology_path: str):
        self.graph = Graph()
        self.graph.parse(ontology_path)

        # Use the exact namespace from your OWL file (area2d_ontology prefix)
        self.AREA = Namespace(
            "http://www.semanticweb.org/user/ontologies/2025/10/area2d_ontology.owl#"
        )

        # Optional debug: show what shapes we find at startup
        shapes = self.list_shapes()
        print("Found shape individuals:", len(shapes))
        for name, uri in shapes:
            print(" -", name, "=>", uri)

    # ------------ Helper methods ------------

    def _label_or_localname(self, uri):
        """Get a human-friendly name: rdfs:label if exists, else localname."""
        g = self.graph
        label = g.value(uri, RDFS.label)
        if label:
            return str(label)
        # fallback: last part after # or /
        text = str(uri)
        if "#" in text:
            return text.split("#")[-1]
        if "/" in text:
            return text.rstrip("/").split("/")[-1]
        return text

    # ------------ Public query methods ------------

    def list_shapes(self):
        """
        Return a list of (display_name, uri) for all Shape individuals.
        Works even if individuals are typed as subclasses like Rectangle, Triangle, etc.
        """
        g = self.graph
        A = self.AREA

        shapes = []

        # 1) Collect Shape class and all of its subclasses
        shape_classes = set([A.Shape])
        for cls in g.subjects(RDFS.subClassOf, A.Shape):
            shape_classes.add(cls)

        # 2) For each such class, collect its individuals
        for cls in shape_classes:
            for s in g.subjects(RDF.type, cls):
                shapes.append((self._label_or_localname(s), s))

        # 3) Deduplicate by URI
        seen = set()
        unique_shapes = []
        for name, uri in shapes:
            if uri not in seen:
                seen.add(uri)
                unique_shapes.append((name, uri))

        # 4) Sort by display name
        unique_shapes.sort(key=lambda x: x[0].lower())
        return unique_shapes

    def get_shape_info(self, shape_uri):
        """
        Given a Shape individual, return a dictionary:
        {
          "name": ...,
          "formulas": [ {"uri": ..., "text": ...}, ... ],
          "skills": [
             {
               "uri": ..., "skillId": ..., "description": ...,
               "difficulty": ..., "prereqs": [..], "misconceptions": [...]
             }
          ]
        }
        """
        g = self.graph
        A = self.AREA

        info = {
            "name": self._label_or_localname(shape_uri),
            "formulas": [],
            "skills": [],
        }

        # --- Formulas linked by hasFormula ---
        for f_uri in g.objects(shape_uri, A.hasFormula):
            formula_text = g.value(f_uri, A.formulaText)
            info["formulas"].append(
                {
                    "uri": f_uri,
                    "text": str(formula_text)
                    if formula_text is not None
                    else "(no formulaText)",
                }
            )

        # --- Skills linked by hasSkill ---
        for s_uri in g.objects(shape_uri, A.hasSkill):
            # In your OWL, the property is skillID (capital D), and some
            # skills use area_circle / area_triangle / etc. instead.
            skill_id_lit = g.value(s_uri, A.skillID)

            # Fallback: check the area_* properties if skillID is not set
            if skill_id_lit is None:
                for prop in [
                    A.area_rectangle,
                    A.area_triangle,
                    A.area_circle,
                    A.area_parallelogram,
                    A.area_composite_shapes,
                ]:
                    v = g.value(s_uri, prop)
                    if v is not None:
                        skill_id_lit = v
                        break

            # Description: use description if present; otherwise fall back to skill_id text
            desc_lit = g.value(s_uri, A.description)
            if desc_lit is None and skill_id_lit is not None:
                desc_lit = skill_id_lit

            # Difficulty
            diff_lit = g.value(s_uri, A.difficultyLevel)
            difficulty = int(diff_lit) if diff_lit is not None else None

            # prerequisites (requiresPrerequisiteSkill)
            prereqs = []
            for pr_uri in g.objects(s_uri, A.requiresPrerequisiteSkill):
                prereqs.append(self._label_or_localname(pr_uri))

            # misconceptions (hasCommonMisconception)
            miss = []
            for m_uri in g.objects(s_uri, A.hasCommonMisconception):
                m_desc = g.value(m_uri, A.description)
                label = self._label_or_localname(m_uri)
                if m_desc:
                    miss.append(f"{label}: {m_desc}")
                else:
                    miss.append(label)

            info["skills"].append(
                {
                    "uri": s_uri,
                    "skillId": str(skill_id_lit)
                    if skill_id_lit is not None
                    else "(no skillID/area_* text)",
                    "description": str(desc_lit)
                    if desc_lit is not None
                    else "(no description)",
                    "difficulty": difficulty,
                    "prereqs": prereqs,
                    "misconceptions": miss,
                }
            )

        return info


class OntologyBrowserApp:
    """
    Simple Tkinter UI that uses AreaOntologyInterface.
    This is your 'User Interface Model' talking to the ontology.
    """

    def __init__(self, root, ontology_path="area2d_ontology.owl"):
        self.root = root
        self.root.title("Area of 2D Shapes Ontology Browser")
        self.root.geometry("800x500")

        try:
            self.onto = AreaOntologyInterface(ontology_path)
        except Exception as e:
            messagebox.showerror("Error loading ontology", str(e))
            raise

        self.shapes = self.onto.list_shapes()
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # Top: shape selection
        top = ttk.Frame(main)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(
            top,
            text="Select a Shape from Ontology:",
            font=("Arial", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.shape_names = [name for name, uri in self.shapes]
        self.shape_combo = ttk.Combobox(
            top,
            values=self.shape_names,
            state="readonly",
            width=40,
        )
        self.shape_combo.grid(row=0, column=1, padx=5)
        if self.shape_names:
            self.shape_combo.current(0)

        show_btn = ttk.Button(top, text="Show Info", command=self.on_show_info)
        show_btn.grid(row=0, column=2, padx=5)

        # Middle: text area for formulas, skills, etc.
        ttk.Label(main, text="Details:", font=("Arial", 11, "bold")).pack(anchor="w")

        self.text = tk.Text(main, wrap="word", height=20)
        self.text.pack(fill="both", expand=True, pady=(5, 0))

        # Scrollbar
        scroll = ttk.Scrollbar(main, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        self._set_intro_text()

    def _set_intro_text(self):
        self.text.delete("1.0", tk.END)
        intro = (
            "This interface is connected to your area-of-2D-shapes ontology.\n\n"
            "Steps:\n"
            "1. Choose a Shape individual from the dropdown.\n"
            "2. Click 'Show Info'.\n\n"
            "You will see:\n"
            "- The formulas linked to that shape (via hasFormula / formulaText).\n"
            "- The skills linked to that shape (via hasSkill).\n"
            "- Any prerequisite skills and common misconceptions.\n\n"
            "This is exactly the kind of domain information your ITS can use\n"
            "to generate hints, choose next tasks, and log which skills the student practiced."
        )
        self.text.insert(tk.END, intro)

    def on_show_info(self):
        if not self.shapes:
            messagebox.showinfo(
                "No shapes", "No Shape individuals were found in the ontology."
            )
            return

        idx = self.shape_combo.current()
        if idx < 0:
            messagebox.showinfo("Select shape", "Please select a shape first.")
            return

        shape_name, shape_uri = self.shapes[idx]
        info = self.onto.get_shape_info(shape_uri)
        self._display_shape_info(info)

    def _display_shape_info(self, info):
        self.text.delete("1.0", tk.END)

        # Header
        self.text.insert(tk.END, f"Shape: {info['name']}\n", ("h1",))
        self.text.insert(tk.END, "-" * 60 + "\n\n")

        # Formulas
        self.text.insert(tk.END, "Formulas:\n", ("h2",))
        if not info["formulas"]:
            self.text.insert(tk.END, "  (No formulas linked via hasFormula)\n\n")
        else:
            for f in info["formulas"]:
                self.text.insert(tk.END, f"  • {f['text']}\n")
            self.text.insert(tk.END, "\n")

        # Skills
        self.text.insert(tk.END, "Skills:\n", ("h2",))
        if not info["skills"]:
            self.text.insert(tk.END, "  (No skills linked via hasSkill)\n")
        else:
            for s in info["skills"]:
                self.text.insert(
                    tk.END,
                    f"\n  Skill text/ID: {s['skillId']}\n"
                    f"  Description: {s['description']}\n",
                )
                if s["difficulty"] is not None:
                    self.text.insert(
                        tk.END, f"  Difficulty level: {s['difficulty']}\n"
                    )
                # Prereqs
                if s["prereqs"]:
                    self.text.insert(tk.END, "  Prerequisite skills:\n")
                    for pr in s["prereqs"]:
                        self.text.insert(tk.END, f"    - {pr}\n")
                else:
                    self.text.insert(
                        tk.END, "  Prerequisite skills: (none)\n"
                    )
                # Misconceptions
                if s["misconceptions"]:
                    self.text.insert(tk.END, "  Common misconceptions:\n")
                    for m in s["misconceptions"]:
                        self.text.insert(tk.END, f"    - {m}\n")
                else:
                    self.text.insert(
                        tk.END, "  Common misconceptions: (none)\n"
                    )

        # Tag styles
        self.text.tag_configure("h1", font=("Arial", 13, "bold"))
        self.text.tag_configure("h2", font=("Arial", 11, "bold"))


def main():
    root = tk.Tk()
    # Change path if your OWL is in another folder
    app = OntologyBrowserApp(root, ontology_path="area2d_ontology.owl")
    root.mainloop()


if __name__ == "__main__":
    main()
