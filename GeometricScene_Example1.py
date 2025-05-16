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
        C1 = Circle(radius=5).move_to(np.array([0, 0, 0]))
        C1.set_stroke(color=RED)
        C1_center = Dot(np.array([0, 0, 0]), color=YELLOW)
        P = Dot(np.array([12, 5, 0]), color=WHITE, radius=0.08)
        L1 = Line(np.array([12, 5, 0]), np.array([-3.55, 3.52, 0]))
        L1.set_stroke(color=YELLOW, width=2)
        L1_tangent_point = Dot(np.array([21.176664702338144, 5.873406029547295, 0]), color=RED, radius=0.08)
        L1_start_dot = Dot(np.array([12, 5, 0]), color=WHITE, radius=0.06)
        L1_end_dot = Dot(np.array([-3.55, 3.52, 0]), color=WHITE, radius=0.06)
        L2 = Line(np.array([12, 5, 0]), np.array([-3.55, 3.52, 0]))
        L2.set_stroke(color=YELLOW, width=2)
        L2_tangent_point = Dot(np.array([21.176664702338144, 5.873406029547295, 0]), color=RED, radius=0.08)
        L2_start_dot = Dot(np.array([12, 5, 0]), color=WHITE, radius=0.06)
        L2_end_dot = Dot(np.array([-3.55, 3.52, 0]), color=WHITE, radius=0.06)
        C1_label = Text('C1 (r=5)', font_size=16).next_to(C1, UP)
        P_label = Text('P', font_size=16).next_to(P, UP+RIGHT)
        L1_label = Text('L1', font_size=16).move_to(np.array([4.225, 4.26, 0]) + np.array([0, 0.5, 0]))
        L2_label = Text('L2', font_size=16).move_to(np.array([4.225, 4.26, 0]) + np.array([0, 0.5, 0]))

        # Animated construction sequence
        self.play(Create(C1), FadeIn(P), Create(L1), Create(L2), run_time=2)

        # Add remaining elements
        self.play(*[FadeIn(obj) for obj in [L1_tangent_point, L2_tangent_point]], run_time=1)
        self.play(*[Write(obj) for obj in [C1_label, P_label, L1_label, L2_label]], run_time=1.5)

        # Final pause
        self.wait(2)
