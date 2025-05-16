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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_CACHE = {}

# Step 1: Input Query Processing
def parse_geometric_description(description):
    # Sanitize input to prevent formatting issues
    sanitized_description = re.sub(r'[{}]', '', description)  # Remove curly braces
    sanitized_description = sanitized_description.replace('\n', ' ').strip()  # Normalize newlines
    
    # Check cache
    cache_key = sanitized_description
    if cache_key in API_CACHE:
        logging.debug(f"Using cached JSON schema for: {cache_key}")
        return API_CACHE[cache_key]
    
#     # Zero-shot examples with detailed tangent placement and geometric principles
#     prompt = f"""You are a geometric parser with expert knowledge of geometric principles. Convert the following natural language description into a JSON schema with 'entities' (shapes, points, lines with properties) and 'relationships' (e.g., tangent, inscribed). Additionally, calculate and include mathematically precise 'positions' for all entities.

# CRITICAL GEOMETRIC PRINCIPLES:
# 1. TANGENT LINE TO CIRCLE: A tangent line to a circle touches the circle at exactly one point. At that contact point, the tangent is perpendicular to the radius at that point.
#    - Given circle center (cx, cy) with radius r, and a point P(px, py) outside the circle, to find the tangent points T:
#      - Calculate d = distance from P to center = sqrt((px-cx)^2 + (py-cy)^2)
#      - The angle of the tangent line θ = asin(r/d)
#      - The tangent points T1 and T2 will be at angles α±θ, where α is the angle from center to point P

# 2. INSCRIBED SHAPES: For regular polygons inscribed in a circle:
#    - Vertices must lie exactly on the circle
#    - For an n-sided polygon, vertices are at angles of 2π*k/n (k=0,1,...n-1)

# 3. TANGENT FROM POINT TO CIRCLE:
#    - For any point P outside a circle, exactly two tangent lines can be drawn
#    - The correct length of a tangent line is sqrt(d^2 - r^2) where d is distance from P to center

# 4. TRIANGLES:
#    - Provide vertices as [x, y] coordinates
#    - For special triangles (equilateral, right), ensure exact measurements

# 5. COORDINATE SYSTEM:
#    - Use a consistent coordinate system with circle centers at origin when possible
#    - Ensure all coordinates are mathematically precise

# OUTPUT REQUIREMENTS:
# For circles: Include center coordinates (x, y) and radius
# For points: Include coordinates (x, y)
# For lines: Include start point (x1, y1) and end point (x2, y2) coordinates
# For polygons: Include vertices as array of points [(x1, y1), (x2, y2), ...]

# ZERO-SHOT EXAMPLE 1 - TANGENT FROM POINT:
# Input: "Draw a circle C1 with center at origin and radius 5. Place point P at coordinates (12,5) and draw tangent lines from P to the circle."
# Output: {{
#   "entities": [
#     {{"type": "circle", "id": "C1", "radius": 5}},
#     {{"type": "point", "id": "P"}},
#     {{"type": "line", "id": "L1"}},
#     {{"type": "line", "id": "L2"}}
#   ],
#   "relationships": [
#     {{"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}},
#     {{"type": "tangent", "from": "L2", "to": "C1", "from_point": "P"}}
#   ],
#   "positions": {{
#     "C1": {{"center": [0, 0], "radius": 5}},
#     "P": {{"point": [12, 5]}},
#     "L1": {{"start": [12, 5], "end": [3.85, 3.21]}},
#     "L2": {{"start": [12, 5], "end": [3.21, -3.85]}}
#   }}
# }}

# ZERO-SHOT EXAMPLE 2 - MULTIPLE CIRCLES WITH TANGENT:
# Input: "Draw two circles C1 and C2 with radii 4 and 3 respectively. Place C1 at origin and C2 at (10,0). Draw a common external tangent line."
# Output: {{
#   "entities": [
#     {{"type": "circle", "id": "C1", "radius": 4}},
#     {{"type": "circle", "id": "C2", "radius": 3}},
#     {{"type": "line", "id": "L1"}}
#   ],
#   "relationships": [
#     {{"type": "tangent", "from": "L1", "to": "C1"}},
#     {{"type": "tangent", "from": "L1", "to": "C2"}}
#   ],
#   "positions": {{
#     "C1": {{"center": [0, 0], "radius": 4}},
#     "C2": {{"center": [10, 0], "radius": 3}},
#     "L1": {{"start": [0, 4], "end": [10, 3]}}
#   }}
# }}

# ZERO-SHOT EXAMPLE 3 - COMPLEX CONSTRUCTION:
# Input: "Draw a circle C with radius 8 centered at origin. Inscribe an equilateral triangle ABC. From point P at (20,10), draw tangents to circle C. Draw a square with side length 5 such that one vertex touches the circle."
# Output: {{
#   "entities": [
#     {{"type": "circle", "id": "C", "radius": 8}},
#     {{"type": "triangle", "id": "ABC"}},
#     {{"type": "point", "id": "P"}},
#     {{"type": "line", "id": "T1"}},
#     {{"type": "line", "id": "T2"}},
#     {{"type": "square", "id": "S", "side_length": 5}}
#   ],
#   "relationships": [
#     {{"type": "inscribed", "shape": "ABC", "in": "C"}},
#     {{"type": "tangent", "from": "T1", "to": "C", "from_point": "P"}},
#     {{"type": "tangent", "from": "T2", "to": "C", "from_point": "P"}},
#     {{"type": "tangent", "from": "S", "to": "C"}}
#   ],
#   "positions": {{
#     "C": {{"center": [0, 0], "radius": 8}},
#     "ABC": {{"vertices": [
#       [8, 0],
#       [-4, 6.93],
#       [-4, -6.93]
#     ]}},
#     "P": {{"point": [20, 10]}},
#     "T1": {{"start": [20, 10], "end": [5.93, 5.33]}},
#     "T2": {{"start": [20, 10], "end": [7.87, -1.38]}},
#     "S": {{"vertices": [
#       [8, 0],
#       [5.54, 5.54],
#       [0, 3.07],
#       [2.46, -2.46]
#     ]}}
#   }}
# }}

# Now, parse this input: "{sanitized_description}"
# Output a JSON schema (only the JSON, no extra text): """

    prompt = f"""You are a geometric parser with expert knowledge of geometric principles. Convert the following natural language description into a JSON schema with 'entities', 'relationships', and mathematically precise 'positions'.

CRITICAL GEOMETRIC PRINCIPLES AND CALCULATIONS:

1. TANGENT LINE TO CIRCLE: 
   - Mathematical definition: A tangent line touches the circle at exactly one point and is perpendicular to the radius at that point.
   - Given a circle with center C(cx,cy) and radius r:
     - For a tangent at angle θ: tangent point T = (cx + r*cos(θ), cy + r*sin(θ))
     - The tangent direction vector is perpendicular to radius: (-sin(θ), cos(θ))
     - For a tangent of length L: endpoint = T + L*(-sin(θ), cos(θ))

2. TANGENT FROM EXTERNAL POINT TO CIRCLE:
   - For point P(px,py) outside a circle with center C(cx,cy) and radius r:
     - Distance d from P to C = sqrt((px-cx)² + (py-cy)²)
     - IMPORTANT: P must be external to circle (d > r)
     - Angle α from C to P = atan2(py-cy, px-cx)
     - Half-angle of tangents θ = asin(r/d)
     - Tangent points T₁ = (cx + r*cos(α+θ), cy + r*sin(α+θ)) and T₂ = (cx + r*cos(α-θ), cy + r*sin(α-θ))
     - Tangent lines are P to T₁ and P to T₂
     - Exact length of each tangent segment = sqrt(d² - r²)

3. COMMON TANGENT TO TWO CIRCLES:
   - For circles C₁(x₁,y₁,r₁) and C₂(x₂,y₂,r₂):
     - External tangents: Construct circle C₃ with center at C₁ and radius (r₁-r₂)
     - Internal tangents: Construct circle C₃ with center at C₁ and radius (r₁+r₂)
     - Calculate tangent points as above

CALCULATION EXAMPLES:

Example 1 - Tangent from Point to Circle:
Circle at (0,0) with radius 2, point P at (5,0):
- Distance d = 5
- Angle α = 0
- θ = asin(2/5) = 0.4115 radians
- Tangent points: T₁ = (2*cos(0.4115), 2*sin(0.4115)) = (1.789, 0.894)
                 T₂ = (2*cos(-0.4115), 2*sin(-0.4115)) = (1.789, -0.894)
- Tangent lines: (5,0) to (1.789, 0.894) and (5,0) to (1.789, -0.894)
- Each tangent length = sqrt(5² - 2²) = sqrt(21) = 4.583

Example 2 - Two Tangents of Specific Length:
Circle at (0,0), radius 2, tangents of length 3:
- For tangent of length 3, point P must be at distance d where sqrt(d² - 2²) = 3
- Therefore d = sqrt(3² + 2²) = sqrt(13) = 3.606
- Place P at (3.606, 0)
- Calculate tangent points as above
- Verify tangent lengths = 3

ZERO-SHOT EXAMPLE 1:
Input: "Draw a circle with radius 3 centered at origin and a point P at (7,4). Draw tangent lines from P to the circle."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 3}},
    {{"type": "point", "id": "P"}},
    {{"type": "line", "id": "L1"}},
    {{"type": "line", "id": "L2"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}},
    {{"type": "tangent", "from": "L2", "to": "C1", "from_point": "P"}}
  ],
  "positions": {{
    "C1": {{"center": [0, 0], "radius": 3}},
    "P": {{"point": [7, 4]}},
    "L1": {{"start": [7, 4], "end": [2.56, 1.46]}},
    "L2": {{"start": [7, 4], "end": [0.81, 2.91]}}
  }}
}}

ZERO-SHOT EXAMPLE 2:
Input: "Draw a circle with radius 2 centered at origin and two tangent lines of length 4 from point P."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 2}},
    {{"type": "point", "id": "P"}},
    {{"type": "line", "id": "L1"}},
    {{"type": "line", "id": "L2"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}},
    {{"type": "tangent", "from": "L2", "to": "C1", "from_point": "P"}}
  ],
  "positions": {{
    "C1": {{"center": [0, 0], "radius": 2}},
    "P": {{"point": [4.47, 0]}},
    "L1": {{"start": [4.47, 0], "end": [2, 1.79]}},
    "L2": {{"start": [4.47, 0], "end": [2, -1.79]}}
  }}
}}

ZERO-SHOT EXAMPLE 3:
Input: "Draw a circle C1 with radius 4 and a circle C2 with radius 2 at (10,0). Draw their common external tangent lines."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 4}},
    {{"type": "circle", "id": "C2", "radius": 2}},
    {{"type": "line", "id": "L1"}},
    {{"type": "line", "id": "L2"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1"}},
    {{"type": "tangent", "from": "L1", "to": "C2"}},
    {{"type": "tangent", "from": "L2", "to": "C1"}},
    {{"type": "tangent", "from": "L2", "to": "C2"}}
  ],
  "positions": {{
    "C1": {{"center": [0, 0], "radius": 4}},
    "C2": {{"center": [10, 0], "radius": 2}},
    "L1": {{"start": [0, 4], "end": [10, 2]}},
    "L2": {{"start": [0, -4], "end": [10, -2]}}
  }}
}}

Now, parse this input: "{sanitized_description}"
Output only the resulting JSON schema (no explanations or calculations):
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gemma2-9b-it",
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
        # json_schema = apply_geometric_corrections(json_schema)
        
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        raise ValueError(f"Failed to parse API response: {str(e)}")
    
    # Cache the result
    API_CACHE[cache_key] = json_schema
    logging.debug(f"Cached JSON schema for: {cache_key}")
    return json_schema




if __name__ == "__main__":
    # Store original description to help with entity extraction later
    description = "draw a circle of radius 2 cm and two tangent of length 3 cm from a single point P."
    
    try:
        print("Processing Example:")
        json_schema = parse_geometric_description(description)
        
        # Add original description to help with value extraction
        json_schema["original_description"] = description

        
        # Print the result
        print(json.dumps(json_schema, indent=2))
        
        # Validate the output
        # Check if we have a circle with radius 2
        has_circle_r2 = any(e["type"] == "circle" and e.get("radius") == 2 for e in json_schema["entities"])
        
        # Check if we have two tangent lines
        tangent_lines = [e for e in json_schema["entities"] if e["type"] == "line"]
        has_two_lines = len(tangent_lines) == 2
        
        # Check if lines are tangent to the circle
        circle_id = next(e["id"] for e in json_schema["entities"] if e["type"] == "circle")
        tangent_relationships = [
            r for r in json_schema["relationships"] 
            if r["type"] == "tangent" and r["to"] == circle_id
        ]
        has_two_tangents = len(tangent_relationships) == 2
        
        # Print validation results
        print("\nValidation Results:")
        print(f"- Has circle with radius 2: {has_circle_r2}")
        print(f"- Has two lines: {has_two_lines}")
        print(f"- Has two tangent relationships: {has_two_tangents}")
        
    except ValueError as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")