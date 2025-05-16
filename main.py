import json
import re
import os
from dotenv import load_dotenv
import logging
from groq import Groq
from code_gen import generate_manim_code
# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Cache for API responses
API_CACHE = {}

def parse_geometric_description(description):
    # Sanitize input
    sanitized_description = re.sub(r'[{}]', '', description).replace('\n', ' ').strip()
    
    # Check cache
    cache_key = sanitized_description
    if cache_key in API_CACHE:
        # logging.debug(f"Using cached JSON schema for: {cache_key}")
        return API_CACHE[cache_key]
    
    # Enhanced prompt with more examples
    prompt = f"""You are a geometric parser with expert knowledge of geometric principles. Convert the following natural language description into a JSON schema with 'entities', 'relationships', and mathematically precise 'positions'. Provide coordinates rounded to four decimal places.

CRITICAL GEOMETRIC PRINCIPLES AND CALCULATIONS:

1. TANGENT FROM EXTERNAL POINT:
   - For circle C(cx,cy) radius r, point P(px,py):
     - Distance d = sqrt((px - cx)^2 + (py - cy)^2)
     - Tangent length L = sqrt(d^2 - r^2)
     - For specified L, d = sqrt(r^2 + L^2)
     - If C at (0,0), P at (d,0): T1 = (r^2/d, r*L/d), T2 = (r^2/d, -r*L/d)

2. INSCRIBED REGULAR POLYGON:
   - n sides, radius r: vertices at [r*cos(2πk/n), r*sin(2πk/n)], k=0 to n-1

3. COMMON TANGENT TO TWO CIRCLES:
   - Centers C1(x1,y1,r1), C2(x2,y2,r2), distance d = sqrt((x2-x1)^2 + (y2-y1)^2)
   - External tangent slope m = ±(r1 - r2)/d if y1 = y2

4. LINE INTERSECTION:
   - Lines y = m1x + b1 and y = m2x + b2: x = (b2 - b1)/(m1 - m2), y = m1x + b1

ZERO-SHOT EXAMPLES:

1. "Draw a circle with radius 2 centered at origin and two tangent lines of length 3 from point P."
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
    "C1": {{"center": [0.0000, 0.0000], "radius": 2}},
    "P": {{"point": [3.6056, 0.0000]}},
    "L1": {{"start": [3.6056, 0.0000], "end": [1.1094, 1.6641]}},
    "L2": {{"start": [3.6056, 0.0000], "end": [1.1094, -1.6641]}}
  }}
}}

2. "Draw a circle with radius 3 and inscribe a regular pentagon."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 3}},
    {{"type": "polygon", "id": "P1", "sides": 5}}
  ],
  "relationships": [
    {{"type": "inscribed", "shape": "P1", "in": "C1"}}
  ],
  "positions": {{
    "C1": {{"center": [0.0000, 0.0000], "radius": 3}},
    "P1": {{"vertices": [
      [3.0000, 0.0000],
      [0.9271, 2.8532],
      [-2.4271, 1.7634],
      [-2.4271, -1.7634],
      [0.9271, -2.8532]
    ]}}
  }}
}}

3. "Draw two circles C1 radius 4 at (0,0) and C2 radius 2 at (10,0). Draw their common external tangents."
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
    "C1": {{"center": [0.0000, 0.0000], "radius": 4}},
    "C2": {{"center": [10.0000, 0.0000], "radius": 2}},
    "L1": {{"start": [-1.0000, 4.0000], "end": [11.0000, 2.0000]}},
    "L2": {{"start": [-1.0000, -4.0000], "end": [11.0000, -2.0000]}}
  }}
}}

4. "Draw a line from (0,0) to (4,4) and another from (0,4) to (4,0). Find their intersection."
Output: {{
  "entities": [
    {{"type": "line", "id": "L1"}},
    {{"type": "line", "id": "L2"}},
    {{"type": "point", "id": "P1"}}
  ],
  "relationships": [
    {{"type": "intersection", "between": ["L1", "L2"], "at": "P1"}}
  ],
  "positions": {{
    "L1": {{"start": [0.0000, 0.0000], "end": [4.0000, 4.0000]}},
    "L2": {{"start": [0.0000, 4.0000], "end": [4.0000, 0.0000]}},
    "P1": {{"point": [2.0000, 2.0000]}}
  }}
}}

5. "Draw a circle radius 5 with a chord of length 8."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 5}},
    {{"type": "line", "id": "L1"}}
  ],
  "relationships": [
    {{"type": "chord", "line": "L1", "in": "C1"}}
  ],
  "positions": {{
    "C1": {{"center": [0.0000, 0.0000], "radius": 5}},
    "L1": {{"start": [-4.0000, 3.0000], "end": [4.0000, 3.0000]}}
  }}
}}

6. "Draw a triangle with vertices at (0,0), (4,0), and (2,3). Inscribe a circle."
Output: {{
  "entities": [
    {{"type": "polygon", "id": "T1", "sides": 3}},
    {{"type": "circle", "id": "C1"}}
  ],
  "relationships": [
    {{"type": "inscribed", "shape": "C1", "in": "T1"}}
  ],
  "positions": {{
    "T1": {{"vertices": [[0.0000, 0.0000], [4.0000, 0.0000], [2.0000, 3.0000]]}},
    "C1": {{"center": [2.0000, 1.0000], "radius": 1.0000}}
  }}
}}

7. "Draw two circles radius 3 at (0,0) and (5,0). Draw their common internal tangents."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 3}},
    {{"type": "circle", "id": "C2", "radius": 3}},
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
    "C1": {{"center": [0.0000, 0.0000], "radius": 3}},
    "C2": {{"center": [5.0000, 0.0000], "radius": 3}},
    "L1": {{"start": [0.0000, 3.0000], "end": [5.0000, 3.0000]}},
    "L2": {{"start": [0.0000, -3.0000], "end": [5.0000, -3.0000]}}
  }}
}}

8. "Draw a circle radius 2 at (0,0) and a line from (-3,3) to (3,-3). Find intersection points."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 2}},
    {{"type": "line", "id": "L1"}},
    {{"type": "point", "id": "P1"}},
    {{"type": "point", "id": "P2"}}
  ],
  "relationships": [
    {{"type": "intersection", "between": ["C1", "L1"], "at": ["P1", "P2"]}}
  ],
  "positions": {{
    "C1": {{"center": [0.0000, 0.0000], "radius": 2}},
    "L1": {{"start": [-3.0000, 3.0000], "end": [3.0000, -3.0000]}},
    "P1": {{"point": [-1.4142, 1.4142]}},
    "P2": {{"point": [1.4142, -1.4142]}}
  }}
}}

9. "Draw a square with side length 4 and circumscribe a circle."
Output: {{
  "entities": [
    {{"type": "polygon", "id": "S1", "sides": 4}},
    {{"type": "circle", "id": "C1"}}
  ],
  "relationships": [
    {{"type": "circumscribed", "shape": "C1", "around": "S1"}}
  ],
  "positions": {{
    "S1": {{"vertices": [[-2.0000, -2.0000], [2.0000, -2.0000], [2.0000, 2.0000], [-2.0000, 2.0000]]}},
    "C1": {{"center": [0.0000, 0.0000], "radius": 2.8284}}
  }}
}}

10. "Draw a circle radius 3 and a point P at (5,5). Draw a line from P tangent to the circle."
Output: {{
  "entities": [
    {{"type": "circle", "id": "C1", "radius": 3}},
    {{"type": "point", "id": "P"}},
    {{"type": "line", "id": "L1"}}
  ],
  "relationships": [
    {{"type": "tangent", "from": "L1", "to": "C1", "from_point": "P"}}
  ],
  "positions": {{
    "C1": {{"center": [0.0000, 0.0000], "radius": 3}},
    "P": {{"point": [5.0000, 5.0000]}},
    "L1": {{"start": [5.0000, 5.0000], "end": [1.8000, 2.4000]}}
  }}
}}

11. "Draw an equilateral triangle with side 6 and inscribe a circle."
Output: {{
  "entities": [
    {{"type": "polygon", "id": "T1", "sides": 3}},
    {{"type": "circle", "id": "C1"}}
  ],
  "relationships": [
    {{"type": "inscribed", "shape": "C1", "in": "T1"}}
  ],
  "positions": {{
    "T1": {{"vertices": [[0.0000, 0.0000], [6.0000, 0.0000], [3.0000, 5.1962]]}},
    "C1": {{"center": [3.0000, 1.7321], "radius": 1.7321}}
  }}
}}

Now, parse this input: "{sanitized_description}"
Output only the resulting JSON schema:
"""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            stream=False,
        )
        json_output = chat_completion.choices[0].message.content
        json_start = json_output.find('{')
        json_end = json_output.rfind('}') + 1
        if json_start == -1 or json_end == 0:
            logging.error(f"No valid JSON found: {json_output}")
            raise ValueError("No valid JSON found")
        json_output = json_output[json_start:json_end]
        json_schema = json.loads(json_output)
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        raise ValueError(f"Failed to parse: {str(e)}")
    
    API_CACHE[cache_key] = json_schema
    # logging.debug(f"Cached JSON schema for: {cache_key}")
    return json_schema

if __name__ == "__main__":
    description = "draw a circle of radius 2 cm and two tangent of length 3 cm from a single point P."
    # description = "draw a circle of radius 2 cm a tangent of length 3 cm from a single point P."
    # description = "Draw an equilateral triangle with side 6 and inscribe a circle."
    try:
        print("Processing:")
        json_schema = parse_geometric_description(description)

        manim_code = generate_manim_code(json_schema)
        with open("GeometricScene.py", "w") as f:
                f.write(manim_code)

        os.system("manim -pql GeometricScene.py GeoScene")


        print(json.dumps(json_schema, indent=2))
    except ValueError as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")