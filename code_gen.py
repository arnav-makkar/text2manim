def generate_manim_code(json_schema):
    """Generate Manim code to render the geometric scene based on computed positions."""
    positions = json_schema["positions"]
    entities = {entity["id"]: entity for entity in json_schema["entities"]}
    relationships = json_schema.get("relationships", [])
    
    code = """from manim import *
import numpy as np

class GeometricScene(Scene):
    def construct(self):
        # Create a title for the scene
        # title = Text("Geometric Construction", font_size=36)
        # title.to_edge(UP)
        # self.add(title)
        
        # Create a coordinate grid for reference
        grid = NumberPlane(
            x_range=[-20, 20, 15],
            y_range=[-15, 15, 15],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 0.5,
                "stroke_opacity": 0.3
            }
        )
        self.add(grid)
        
        # Create objects
"""
    # Track entities for later use in relationships
    entity_objects = {}
    
    # Create entities
    for entity_id, pos in positions.items():
        entity_type = entities[entity_id]["type"]
        
        if entity_type == "circle":
            center = pos["center"]
            radius = pos["radius"]
            code += f"        {entity_id} = Circle(radius={radius}).move_to(np.array([{center[0]}, {center[1]}, 0]))\n"
            code += f"        {entity_id}.set_stroke(color=BLUE)\n"
            entity_objects[entity_id] = {"type": "circle", "manim_obj": f"{entity_id}"}
        
        elif entity_type == "point":
            point = pos["point"]
            code += f"        {entity_id} = Dot(np.array([{point[0]}, {point[1]}, 0]), color=WHITE)\n"
            entity_objects[entity_id] = {"type": "point", "manim_obj": f"{entity_id}"}
        
        elif entity_type == "line":
            start = pos["start"]
            end = pos["end"]
            code += f"        {entity_id} = Line(np.array([{start[0]}, {start[1]}, 0]), np.array([{end[0]}, {end[1]}, 0]))\n"
            code += f"        {entity_id}.set_stroke(color=YELLOW)\n"
            entity_objects[entity_id] = {"type": "line", "manim_obj": f"{entity_id}"}
        
        elif entity_type == "polygon":
            vertices = pos["vertices"]
            sides = entities[entity_id]["sides"]
            
            # Convert vertices to the required format for Manim
            vertices_str = ", ".join(f"np.array([{v[0]}, {v[1]}, 0])" for v in vertices)
            
            # Color mapping based on polygon type
            colors = {
                3: "GREEN",     # Triangle
                4: "PURPLE",    # Square/Rectangle
                5: "TEAL",      # Pentagon
                6: "GOLD",      # Hexagon
            }
            color = colors.get(sides, "WHITE")
            
            # Create the polygon
            code += f"        {entity_id}_vertices = [{vertices_str}]\n"
            code += f"        {entity_id} = Polygon(*{entity_id}_vertices, color={color})\n"
            
            # Add special name if it's a known regular polygon
            if sides == 3:
                polygon_name = "Triangle"
            elif sides == 4:
                # Check if it's a square (all sides equal)
                is_square = True
                for i in range(sides):
                    next_i = (i + 1) % sides
                    current_side = [vertices[i][0] - vertices[next_i][0], vertices[i][1] - vertices[next_i][1]]
                    current_length = (current_side[0]**2 + current_side[1]**2)**0.5
                    
                    if i > 0:
                        if abs(current_length - prev_length) > 0.001:
                            is_square = False
                            break
                    prev_length = current_length
                
                polygon_name = "Square" if is_square else "Rectangle"
            elif sides == 5:
                polygon_name = "Pentagon"
            elif sides == 6:
                polygon_name = "Hexagon"
            else:
                polygon_name = f"{sides}-gon"
                
            # Get unit if specified
            unit = entities[entity_id].get("unit", "")
            unit_str = f" ({unit})" if unit else ""
            
            # Add comment about the polygon type
            code += f"        # This is a {polygon_name}{unit_str}\n"
            entity_objects[entity_id] = {"type": "polygon", "manim_obj": f"{entity_id}"}
    
    # Define a variable to track if we've created relationship visualizations
    has_inscribed_relations = False
    
    # Handle relationships (inscribed circles, etc.)
    for rel in relationships:
        if rel["type"] == "inscribed":
            shape_id = rel["shape"]
            in_shape_id = rel["in"]
            
            if entities[shape_id]["type"] == "circle" and entities[in_shape_id]["type"] == "polygon":
                # Highlight the inscribed relationship
                has_inscribed_relations = True
                code += f"        # Highlight the inscribed relationship between {shape_id} and {in_shape_id}\n"
                code += f"        inscribed_relation_{shape_id}_{in_shape_id} = DashedVMobject({shape_id}, num_dashes=15)\n"
                code += f"        inscribed_relation_{shape_id}_{in_shape_id}.set_stroke(opacity=0.7, color=RED)\n"
    
    # Add animations for creating entities in logical order
    point_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "point"]
    polygon_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "polygon"]
    circle_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "circle"]
    line_ids = [entity["id"] for entity in json_schema["entities"] if entity["type"] == "line"]
    
    # Add points first
    for pid in point_ids:
        code += f"        self.play(FadeIn({pid}))\n"
    
    # Then add polygons
    for poly_id in polygon_ids:
        code += f"        self.play(Create({poly_id}))\n"
    
    # Then add circles
    for cid in circle_ids:
        code += f"        self.play(Create({cid}))\n"
    
    # Finally add lines (including tangent lines)
    for lid in line_ids:
        code += f"        self.play(Create({lid}))\n"
    
    # Add relationship visualizations
    for rel in relationships:
        if rel["type"] == "inscribed":
            shape_id = rel["shape"]
            in_shape_id = rel["in"]
            if entities[shape_id]["type"] == "circle" and entities[in_shape_id]["type"] == "polygon":
                code += f"        self.play(Create(inscribed_relation_{shape_id}_{in_shape_id}))\n"
    
    # Add labels to all entities
    for entity_id, pos in positions.items():
        entity_type = entities[entity_id]["type"]
        
        if entity_type == "circle":
            center = pos["center"]
            code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).next_to({entity_id}, UP)\n"
            code += f"        self.add({entity_id}_label)\n"
            
        elif entity_type == "point":
            code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).next_to({entity_id}, RIGHT)\n"
            code += f"        self.add({entity_id}_label)\n"
            
        elif entity_type == "line":
            start = pos["start"]
            end = pos["end"]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).move_to(np.array([{mid_x}, {mid_y}, 0]) + np.array([0, 0.2, 0]))\n"
            code += f"        self.add({entity_id}_label)\n"
            
            # Add tangent point if it's a tangent line
            if any(rel["type"] == "tangent" and rel["from"] == entity_id for rel in relationships):
                code += f"        tangent_point = Dot(np.array([{end[0]}, {end[1]}, 0]), color=RED)\n"
                code += f"        self.play(FadeIn(tangent_point))\n"
                
        elif entity_type == "polygon":
            # Calculate center of polygon for label placement
            vertices = pos["vertices"]
            center_x = sum(v[0] for v in vertices) / len(vertices)
            center_y = sum(v[1] for v in vertices) / len(vertices)
            code += f"        {entity_id}_label = Text('{entity_id}', font_size=24).move_to(np.array([{center_x}, {center_y}, 0]))\n"
            code += f"        self.add({entity_id}_label)\n"
    
    code += "        self.wait(2)\n"
    return code


# Example usage with the provided schemas
if __name__ == "__main__":
    # Test with square example
    square_schema = {
        "entities": [
            {
                "type": "polygon",
                "id": "S1",
                "sides": 4,
                "unit": "cm"
            }
        ],
        "relationships": [],
        "positions": {
            "S1": {
                "vertices": [
                    [-1.5, -1.5],
                    [1.5, -1.5],
                    [1.5, 1.5],
                    [-1.5, 1.5]
                ]
            }
        }
    }
    
    # Test with triangle and inscribed circle example
    triangle_circle_schema = {
        "entities": [
            {
                "type": "polygon",
                "id": "T1",
                "sides": 3
            },
            {
                "type": "circle",
                "id": "C1"
            }
        ],
        "relationships": [
            {
                "type": "inscribed",
                "shape": "C1",
                "in": "T1"
            }
        ],
        "positions": {
            "T1": {
                "vertices": [
                    [0.0, 0.0],
                    [6.0, 0.0],
                    [3.0, 5.1962]
                ]
            },
            "C1": {
                "center": [3.0, 1.7321],
                "radius": 1.7321
            }
        }
    }
    
    # Generate code for each schema
    square_code = generate_manim_code(square_schema)
    with open("GeometricScene_Square.py", "w") as f:
        f.write(square_code)
    
    triangle_circle_code = generate_manim_code(triangle_circle_schema)
    with open("GeometricScene_TriangleCircle.py", "w") as f:
        f.write(triangle_circle_code)
    
    # Print the code for the second example
    print(triangle_circle_code)