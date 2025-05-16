def generate_manim_code(json_schema):
    """Generate Manim code to render the geometric scene based on computed positions."""
    positions = json_schema["positions"]
    entities = {entity["id"]: entity for entity in json_schema["entities"]}
    
    code = """from manim import *
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
"""
    # Create entities
    for entity_id, pos in positions.items():
        entity_type = entities[entity_id]["type"]
        if entity_type == "circle":
            center = pos["center"]
            radius = pos["radius"]
            code += f"        {entity_id} = Circle(radius={radius}).move_to(np.array([{center[0]}, {center[1]}, 0]))\n"
            code += f"        {entity_id}.set_stroke(color=BLUE)\n"
        elif entity_type == "point":
            point = pos["point"]
            code += f"        {entity_id} = Dot(np.array([{point[0]}, {point[1]}, 0]), color=WHITE)\n"
        elif entity_type == "line":
            start = pos["start"]
            end = pos["end"]
            code += f"        {entity_id} = Line(np.array([{start[0]}, {start[1]}, 0]), np.array([{end[0]}, {end[1]}, 0]))\n"
            code += f"        {entity_id}.set_stroke(color=YELLOW)\n"
    
    # Add animations for creating entities
    circle_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "circle"]
    point_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "point"]
    line_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "line"]
    
    for cid in circle_ids:
        code += f"        self.play(Create({cid}))\n"
    for pid in point_ids:
        code += f"        self.play(FadeIn({pid}))\n"
    for lid in line_ids:
        code += f"        self.play(Create({lid}))\n"
    
    # Add labels and tangent points
    for entity in json_schema["entities"]:
        entity_id = entity["id"]
        if entity_id in positions:
            if entity["type"] == "circle":
                code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).next_to({entity_id}, UP)\n"
                code += f"        self.add({entity_id}_label)\n"
            elif entity["type"] == "point":
                code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).next_to({entity_id}, RIGHT)\n"
                code += f"        self.add({entity_id}_label)\n"
            elif entity["type"] == "line":
                start = positions[entity_id]["start"]
                end = positions[entity_id]["end"]
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).move_to(np.array([{mid_x}, {mid_y}, 0]) + np.array([0, 0.2, 0]))\n"
                code += f"        self.add({entity_id}_label)\n"
                # Add tangent point if it's a tangent line
                if any(rel["type"] == "tangent" and rel["from"] == entity_id for rel in json_schema.get("relationships", [])):
                    code += f"        tangent_point = Dot(np.array([{end[0]}, {end[1]}, 0]), color=RED)\n"
                    code += f"        self.play(FadeIn(tangent_point))\n"
    
    code += "        self.wait(2)\n"
    return code

# Example usage with the provided schema
schema = {
  "entities": [
    {"type": "circle", "id": "C1", "radius": 2},
    {"type": "point", "id": "P"},
    {"type": "line", "id": "L1"}
  ],
  "relationships": [
    {"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}
  ],
  "positions": {
    "C1": {"center": [0.0, 0.0], "radius": 2},
    "P": {"point": [3.6056, 0.0]},
    "L1": {"start": [3.6056, 0.0], "end": [1.1094, 1.6641]}
  }
}

if __name__ == "__main__":
    manim_code = generate_manim_code(schema)
    with open("GeometricScene_Example1.py", "w") as f:
            f.write(manim_code)
    print(manim_code)