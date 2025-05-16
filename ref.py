import json
import re
import requests
import os
import numpy as np
from dotenv import load_dotenv
from manim import *
import logging
import math
from groq import Groq

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
# Store your OpenRouter API key in .env as: OPENROUTER_API_KEY=your-api-key-here
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not OPENROUTER_API_KEY:
#     raise ValueError("OPENROUTER_API_KEY not found in .env file. Create .env with OPENROUTER_API_KEY=your-api-key-here")

# Cache for API responses to speed up development
API_CACHE = {}

# Step 1: Input Query Processing
def parse_geometric_description(description):
    """Parse natural language into a structured JSON schema using OpenRouter API (Qwen)."""
    # Sanitize input to prevent formatting issues
    sanitized_description = re.sub(r'[{}]', '', description)  # Remove curly braces
    sanitized_description = sanitized_description.replace('\n', ' ').strip()  # Normalize newlines
    
    # Check cache
    cache_key = sanitized_description
    if cache_key in API_CACHE:
        logging.debug(f"Using cached JSON schema for: {cache_key}")
        return API_CACHE[cache_key]
    
    # Zero-shot examples with detailed tangent placement and geometric principles
    prompt = f"""You are a geometric parser with expert knowledge of geometric principles. Convert the following natural language description into a JSON schema with 'entities' (shapes, points, lines with properties) and 'relationships' (e.g., tangent, inscribed). Additionally, calculate and include mathematically precise 'positions' for all entities.

CRITICAL GEOMETRIC PRINCIPLES:
1. TANGENT LINE TO CIRCLE: A tangent line to a circle touches the circle at exactly one point. At that contact point, the tangent is perpendicular to the radius at that point.
   - Given circle center (cx, cy) with radius r, and a point P(px, py) outside the circle, to find the tangent points T:
     - Calculate d = distance from P to center = sqrt((px-cx)^2 + (py-cy)^2)
     - The angle of the tangent line θ = asin(r/d)
     - The tangent points T1 and T2 will be at angles α±θ, where α is the angle from center to point P

2. INSCRIBED SHAPES: For regular polygons inscribed in a circle:
   - Vertices must lie exactly on the circle
   - For an n-sided polygon, vertices are at angles of 2π*k/n (k=0,1,...n-1)

3. TANGENT FROM POINT TO CIRCLE:
   - For any point P outside a circle, exactly two tangent lines can be drawn
   - The correct length of a tangent line is sqrt(d^2 - r^2) where d is distance from P to center

4. TRIANGLES:
   - Provide vertices as [x, y] coordinates
   - For special triangles (equilateral, right), ensure exact measurements

5. COORDINATE SYSTEM:
   - Use a consistent coordinate system with circle centers at origin when possible
   - Ensure all coordinates are mathematically precise

OUTPUT REQUIREMENTS:
For circles: Include center coordinates (x, y) and radius
For points: Include coordinates (x, y)
For lines: Include start point (x1, y1) and end point (x2, y2) coordinates
For polygons: Include vertices as array of points [(x1, y1), (x2, y2), ...]

ZERO-SHOT EXAMPLE 1 - TANGENT FROM POINT:
Input: "Draw a circle C1 with center at origin and radius 5. Place point P at coordinates (12,5) and draw tangent lines from P to the circle."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 5}},
    {{"type": "point", "id": "P"}},
    {{"type": "line", "id": "L1"}},
    {{"type": "line", "id": "L2"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}},
    {{"type": "tangent", "from": "L2", "to": "C1", "from_point": "P"}}
  ],
  "positions": {{
    "C1": {{"center": [0, 0], "radius": 5}},
    "P": {{"point": [12, 5]}},
    "L1": {{"start": [12, 5], "end": [3.85, 3.21]}},
    "L2": {{"start": [12, 5], "end": [3.21, -3.85]}}
  }}
}}

ZERO-SHOT EXAMPLE 2 - MULTIPLE CIRCLES WITH TANGENT:
Input: "Draw two circles C1 and C2 with radii 4 and 3 respectively. Place C1 at origin and C2 at (10,0). Draw a common external tangent line."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 4}},
    {{"type": "circle", "id": "C2", "radius": 3}},
    {{"type": "line", "id": "L1"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1"}},
    {{"type": "tangent", "from": "L1", "to": "C2"}}
  ],
  "positions": {{
    "C1": {{"center": [0, 0], "radius": 4}},
    "C2": {{"center": [10, 0], "radius": 3}},
    "L1": {{"start": [0, 4], "end": [10, 3]}}
  }}
}}

ZERO-SHOT EXAMPLE 3 - COMPLEX CONSTRUCTION:
Input: "Draw a circle C with radius 8 centered at origin. Inscribe an equilateral triangle ABC. From point P at (20,10), draw tangents to circle C. Draw a square with side length 5 such that one vertex touches the circle."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C", "radius": 8}},
    {{"type": "triangle", "id": "ABC"}},
    {{"type": "point", "id": "P"}},
    {{"type": "line", "id": "T1"}},
    {{"type": "line", "id": "T2"}},
    {{"type": "square", "id": "S", "side_length": 5}}
  ],
  "relationships": [
    {{"type": "inscribed", "shape": "ABC", "in": "C"}},
    {{"type": "tangent", "from": "T1", "to": "C", "from_point": "P"}},
    {{"type": "tangent", "from": "T2", "to": "C", "from_point": "P"}},
    {{"type": "tangent", "from": "S", "to": "C"}}
  ],
  "positions": {{
    "C": {{"center": [0, 0], "radius": 8}},
    "ABC": {{"vertices": [
      [8, 0],
      [-4, 6.93],
      [-4, -6.93]
    ]}},
    "P": {{"point": [20, 10]}},
    "T1": {{"start": [20, 10], "end": [5.93, 5.33]}},
    "T2": {{"start": [20, 10], "end": [7.87, -1.38]}},
    "S": {{"vertices": [
      [8, 0],
      [5.54, 5.54],
      [0, 3.07],
      [2.46, -2.46]
    ]}}
  }}
}}

Now, parse this input: "{sanitized_description}"
Output a JSON schema (only the JSON, no extra text): """

    try:
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # Make API call
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="qwen-qwq-32b",
            stream=False,
        )
        
        # Log the full response for debugging
        response_data = chat_completion.to_dict()
        logging.debug(f"Full Groq API response: {response_data}")
        
        # Extract the content
        json_output = chat_completion.choices[0].message.content
        
        # Parse JSON output
        json_start = json_output.find('{')
        json_end = json_output.rfind('}') + 1
        # if json_start Ascertain whether the output is valid JSON
        if json_start == -1 or json_end == 0:
            logging.error(f"No valid JSON found in API response: {json_output}")
            raise ValueError("No valid JSON found in API response")
        json_output = json_output[json_start:json_end]
        
        json_schema = json.loads(json_output)
        
        # Post-process positions to apply mathematical correctness
        json_schema = apply_geometric_corrections(json_schema)
        
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        raise ValueError(f"Failed to parse API response: {str(e)}")
    
    # Cache the result
    API_CACHE[cache_key] = json_schema
    logging.debug(f"Cached JSON schema for: {cache_key}")
    return json_schema

def apply_geometric_corrections(json_schema):
    """Apply mathematical corrections to ensure geometric validity."""
    
    # Process each relationship to ensure geometric correctness
    for relationship in json_schema.get("relationships", []):
        if relationship["type"] == "tangent":
            from_id = relationship.get("from")
            to_id = relationship.get("to")
            from_point_id = relationship.get("from_point")
            
            # Correct tangent lines from point to circle
            if from_point_id and to_id and from_id:
                if (from_point_id in json_schema["positions"] and 
                    to_id in json_schema["positions"] and 
                    from_id in json_schema["positions"]):
                    
                    # Get point coordinates
                    point_pos = json_schema["positions"][from_point_id].get("point")
                    
                    # Get circle center and radius
                    circle_center = json_schema["positions"][to_id].get("center")
                    circle_radius = json_schema["positions"][to_id].get("radius")
                    
                    if point_pos and circle_center and circle_radius:
                        # Correct the tangent points using proper geometry
                        json_schema["positions"][from_id] = calculate_tangent_from_point_to_circle(
                            point_pos, circle_center, circle_radius, from_id
                        )
    
    return json_schema

def calculate_tangent_from_point_to_circle(point, center, radius, line_id):
    """Calculate accurate tangent points from an external point to a circle."""
    px, py = point
    cx, cy = center
    r = radius
    
    # Distance from point to center
    d = math.sqrt((px - cx)**2 + (py - cy)**2)
    
    # Basic validation
    if d <= r:
        logging.warning(f"Point {point} is inside or on the circle with center {center} and radius {r}")
        return {"start": point, "end": [cx + r, cy]}  # Default fallback
    
    # Distance from center to tangent point on circle
    # Using Pythagoras: r^2 + h^2 = d^2, where h is from center to tangent line
    h = math.sqrt(d**2 - r**2)
    
    # Angle from center to point
    alpha = math.atan2(py - cy, px - cx)
    
    # Angle to tangent points (using inverse sine)
    theta = math.asin(r / d)
    
    # Calculate first tangent point
    angle1 = alpha + theta
    tx1 = cx + r * math.cos(angle1 + math.pi/2)
    ty1 = cy + r * math.sin(angle1 + math.pi/2)
    
    # Calculate second tangent point
    angle2 = alpha - theta
    tx2 = cx + r * math.cos(angle2 - math.pi/2)
    ty2 = cy + r * math.sin(angle2 - math.pi/2)
    
    # For a single line, we'll use the first tangent point
    # If there are multiple tangent lines, they should be handled by the main algorithm
    return {"start": point, "end": [round(tx1, 2), round(ty1, 2)]}

# Step 2: Code Generation with Enhanced Visuals
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
    # Create a list to track which entities should be added
    entities_to_add = []
    labels_to_add = []
    tangent_points_to_add = []
    animations = []

    # Add entities with proper animation
    for entity_id, pos in positions.items():
        entity_info = entities.get(entity_id, {})
        entity_type = entity_info.get("type", "")
        
        if "center" in pos:  # Circle
            x, y = pos["center"]
            r = pos["radius"]
            code += f"        {entity_id} = Circle(radius={r}).move_to(np.array([{x}, {y}, 0]))\n"
            code += f"        {entity_id}.set_stroke(color=RED)\n"
            code += f"        {entity_id}_center = Dot(np.array([{x}, {y}, 0]), color=YELLOW)\n"
            entities_to_add.append(entity_id)
            entities_to_add.append(f"{entity_id}_center")
            animations.append(f"Create({entity_id})")
        
        elif "vertices" in pos:  # Polygon (e.g., Square, Triangle)
            vertices = ", ".join([f"np.array([{v[0]}, {v[1]}, 0])" for v in pos["vertices"]])
            if entity_type.lower() == "triangle":
                code += f"        {entity_id} = Polygon({vertices})\n"
                code += f"        {entity_id}.set_stroke(color=GREEN)\n"
                code += f"        {entity_id}.set_fill(GREEN, opacity=0.2)\n"
            else:  # Square or other polygon
                code += f"        {entity_id} = Polygon({vertices})\n"
                code += f"        {entity_id}.set_stroke(color=BLUE)\n"
                code += f"        {entity_id}.set_fill(BLUE, opacity=0.2)\n"
            
            # Add dots for vertices
            for i, v in enumerate(pos["vertices"]):
                code += f"        {entity_id}_v{i} = Dot(np.array([{v[0]}, {v[1]}, 0]), color=YELLOW, radius=0.05)\n"
                entities_to_add.append(f"{entity_id}_v{i}")
            
            entities_to_add.append(entity_id)
            animations.append(f"Create({entity_id})")
        
        elif "start" in pos and "end" in pos:  # Line
            start_x, start_y = pos["start"]
            end_x, end_y = pos["end"]
            
            # Check if this is a tangent line
            is_tangent = False
            tangent_point = None
            for rel in json_schema.get("relationships", []):
                if rel.get("type") == "tangent" and rel.get("from") == entity_id:
                    is_tangent = True
                    # Find the circle this line is tangent to
                    circle_id = rel.get("to")
                    if circle_id in positions and "center" in positions[circle_id]:
                        # Calculate the exact tangent point on the circle
                        circle_center = positions[circle_id]["center"]
                        circle_radius = positions[circle_id]["radius"]
                        # Call helper function to find tangent point precisely
                        tangent_point = find_tangent_point(
                            [start_x, start_y], [end_x, end_y], 
                            circle_center, circle_radius
                        )
            
            code += f"        {entity_id} = Line(np.array([{start_x}, {start_y}, 0]), np.array([{end_x}, {end_y}, 0]))\n"
            
            if is_tangent:
                code += f"        {entity_id}.set_stroke(color=YELLOW, width=2)\n"
                # Add a dot at the calculated tangent point if available
                if tangent_point:
                    tx, ty = tangent_point
                    code += f"        {entity_id}_tangent_point = Dot(np.array([{tx}, {ty}, 0]), color=RED, radius=0.08)\n"
                    tangent_points_to_add.append(f"{entity_id}_tangent_point")
            else:
                code += f"        {entity_id}.set_stroke(color=WHITE)\n"
            
            # Add dots at start and end of line
            code += f"        {entity_id}_start_dot = Dot(np.array([{start_x}, {start_y}, 0]), color=WHITE, radius=0.06)\n"
            code += f"        {entity_id}_end_dot = Dot(np.array([{end_x}, {end_y}, 0]), color=WHITE, radius=0.06)\n"
            
            entities_to_add.append(entity_id)
            entities_to_add.append(f"{entity_id}_start_dot")
            entities_to_add.append(f"{entity_id}_end_dot")
            animations.append(f"Create({entity_id})")
        
        elif "point" in pos:  # Point
            x, y = pos["point"]
            code += f"        {entity_id} = Dot(np.array([{x}, {y}, 0]), color=WHITE, radius=0.08)\n"
            entities_to_add.append(entity_id)
            animations.append(f"FadeIn({entity_id})")
    
    # Add labels for entities
    for entity in json_schema["entities"]:
        entity_id = entity["id"]
        if entity_id in positions:
            label_text = f"{entity_id}"
            if "radius" in entity:
                label_text += f" (r={entity['radius']}{entity.get('unit', '')})"
            elif "length" in entity:
                label_text += f" (l={entity['length']}{entity.get('unit', '')})"
                
            if "center" in positions[entity_id]:
                x, y = positions[entity_id]["center"]
                code += f"        {entity_id}_label = Text('{label_text}', font_size=16).next_to({entity_id}, UP)\n"
                labels_to_add.append(f"{entity_id}_label")
            elif "point" in positions[entity_id]:
                x, y = positions[entity_id]["point"]
                code += f"        {entity_id}_label = Text('{label_text}', font_size=16).next_to({entity_id}, UP+RIGHT)\n"
                labels_to_add.append(f"{entity_id}_label")
            elif "start" in positions[entity_id]:
                # Place label near the middle of the line
                start_x, start_y = positions[entity_id]["start"]
                end_x, end_y = positions[entity_id]["end"]
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                code += f"        {entity_id}_label = Text('{label_text}', font_size=16).move_to(np.array([{mid_x}, {mid_y}, 0]) + np.array([0, 0.5, 0]))\n"
                labels_to_add.append(f"{entity_id}_label")
    
    # Add animated construction sequence
    code += "\n        # Animated construction sequence\n"
    if animations:
        code += f"        self.play({', '.join(animations)}, run_time=2)\n"
    
    # Add all objects with proper layering
    code += "\n        # Add remaining elements\n"
    
    # Add tangent points after the main elements for emphasis
    if tangent_points_to_add:
        code += f"        self.play(*[FadeIn(obj) for obj in [{', '.join(tangent_points_to_add)}]], run_time=1)\n"
    
    # Add labels at the end
    if labels_to_add:
        code += f"        self.play(*[Write(obj) for obj in [{', '.join(labels_to_add)}]], run_time=1.5)\n"
    
    code += "\n        # Final pause\n"
    code += "        self.wait(2)\n"
    
    # Log generated code for debugging
    logging.debug(f"Generated Manim code:\n{code}")
    return code

def find_tangent_point(line_start, line_end, circle_center, circle_radius):
    """Find the exact point where a line is tangent to a circle."""
    # Convert inputs to numpy arrays for vector operations
    start = np.array(line_start)
    end = np.array(line_end)
    center = np.array(circle_center)
    
    # Direction vector of the line
    line_dir = end - start
    line_dir = line_dir / np.linalg.norm(line_dir)  # Normalize
    
    # Vector from start to center
    start_to_center = center - start
    
    # Project start_to_center onto line_dir to find closest point on line to circle center
    proj_length = np.dot(start_to_center, line_dir)
    closest_point = start + proj_length * line_dir
    
    # Vector from closest point to center
    closest_to_center = center - closest_point
    dist = np.linalg.norm(closest_to_center)
    
    # If the distance is approximately equal to the radius, we found our tangent point
    if abs(dist - circle_radius) < 0.01:
        return closest_point.tolist()
    
    # Otherwise, we need to find where the line intersects the circle
    # This is a bit complex but necessary for accuracy
    # Vector from start to center
    v = center - start
    
    # Coefficients for quadratic equation
    a = np.dot(line_dir, line_dir)
    b = 2 * np.dot(v, line_dir)
    c = np.dot(v, v) - circle_radius**2
    
    # Discriminant
    disc = b**2 - 4*a*c
    
    # If disc < 0, no intersection
    if disc < 0:
        return end.tolist()  # Default to end point if no solution
    
    # Calculate the two possible intersection points
    t1 = (-b + math.sqrt(disc)) / (2*a)
    t2 = (-b - math.sqrt(disc)) / (2*a)
    
    # Choose the intersection point that lies on our line segment
    point1 = start + t1 * line_dir
    point2 = start + t2 * line_dir
    
    # Check which point is between start and end
    line_length = np.linalg.norm(end - start)
    dist1_start = np.linalg.norm(point1 - start)
    dist1_end = np.linalg.norm(point1 - end)
    dist2_start = np.linalg.norm(point2 - start)
    dist2_end = np.linalg.norm(point2 - end)
    
    # Choose the point that's closest to being on the line segment
    if abs(dist1_start + dist1_end - line_length) < abs(dist2_start + dist2_end - line_length):
        return point1.tolist()
    else:
        return point2.tolist()

# Example usage
if __name__ == "__main__":
    # Example 1: Basic Tangent Case
    description1 = "Draw a circle with center at origin and radius 5. Draw a tangent to the circle from point P at (12,5)."
    # description1 = "Draw two circles C1 and C2 with radii 4 and 3 respectively. Place C1 at origin and C2 at (10,0). Draw a common external tangent line."

    
    # Example 2: Complex case with multiple tangents and shapes
    # description2 = "Draw a circle C1 with radius 8 at origin. Draw another circle C2 with radius 4 at (15,0). Draw the common external tangent lines. Also inscribe an equilateral triangle in C1."
    
    # Example 3: Very complex case with various geometric elements
    # description3 = "Draw a circle with radius 6 at origin. Place point P at (12,8). Draw two tangent lines from P to the circle. Inscribe a square in the circle. Draw another circle with radius 3 tangent to the first circle."
    
    try:
        print("Processing Example 1:")
        json_schema1 = parse_geometric_description(description1)
        print(json.dumps(json_schema1, indent=2))  # For debugging
        manim_code1 = generate_manim_code(json_schema1)
        with open("GeometricScene_Example1.py", "w") as f:
            f.write(manim_code1)
        print("Generated Manim code saved to GeometricScene_Example1.py")
        
        # print("\nProcessing Example 2:")
        # json_schema2 = parse_geometric_description(description2)
        # print(json.dumps(json_schema2, indent=2))  # For debugging
        # manim_code2 = generate_manim_code(json_schema2)
        # with open("GeometricScene_Example2.py", "w") as f:
        #     f.write(manim_code2)
        # print("Generated Manim code saved to GeometricScene_Example2.py")
        
        # print("\nProcessing Example 3:")
        # json_schema3 = parse_geometric_description(description3)
        # print(json.dumps(json_schema3, indent=2))  # For debugging
        # manim_code3 = generate_manim_code(json_schema3)
        # with open("GeometricScene_Example3.py", "w") as f:
        #     f.write(manim_code3)
        # print("Generated Manim code saved to GeometricScene_Example3.py")
        
        print("\nAll examples processed successfully!")
    except ValueError as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")