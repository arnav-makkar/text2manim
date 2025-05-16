from manim import *
import numpy as np

class GeometricScene(Scene):
    def construct(self):
        # Create a title for the scene
        title = Text("Geometric Construction", font_size=36)
        title.to_edge(UP)
        self.add(title)
        
        # Create a coordinate grid for reference
        grid = NumberPlane(
            x_range=[-20, 20, 5],
            y_range=[-15, 15, 5],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 0.5,
                "stroke_opacity": 0.3
            }
        )
        self.add(grid)
        
        # Create objects
        C1 = Circle(radius=2).move_to(np.array([0.0, 0.0, 0]))
        C1.set_stroke(color=BLUE)
        P = Dot(np.array([3.0, 0.0, 0]), color=WHITE)
        L1 = Line(np.array([3.0, 0.0, 0]), np.array([1.0, 1.299, 0]))
        L1.set_stroke(color=YELLOW)
        L2 = Line(np.array([3.0, 0.0, 0]), np.array([1.0, -1.299, 0]))
        L2.set_stroke(color=YELLOW)
        self.play(Create(C1))
        self.play(FadeIn(P))
        self.play(Create(L1))
        self.play(Create(L2))
        C1_label = Text('C1', font_size=24).next_to(C1, UP)
        self.add(C1_label)
        P_label = Text('P', font_size=24).next_to(P, RIGHT)
        self.add(P_label)
        L1_label = Text('L1', font_size=24).move_to(np.array([2.0, 0.6495, 0]) + np.array([0, 0.2, 0]))
        self.add(L1_label)
        tangent_point = Dot(np.array([1.0, 1.299, 0]), color=RED)
        self.play(FadeIn(tangent_point))
        L2_label = Text('L2', font_size=24).move_to(np.array([2.0, -0.6495, 0]) + np.array([0, 0.2, 0]))
        self.add(L2_label)
        tangent_point = Dot(np.array([1.0, -1.299, 0]), color=RED)
        self.play(FadeIn(tangent_point))
        self.wait(2)
