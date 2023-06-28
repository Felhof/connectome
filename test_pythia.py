# %%
from transformer_lens import HookedTransformer
import torch
from IPython.display import SVG
import transformer_lens

import core
# %%
torch.set_grad_enabled(False)
# %%
pythia = HookedTransformer.from_pretrained("pythia-2.8b", device="cpu")
# %%
pythia.to("cuda")
# %%
CODE = """
from typing import List
from math import pi

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, bottom_left: Point, top_right: Point) -> None:
        self.bottom_left = bottom_left
        self.top_right = top_right
        
class Circle:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

class Polygon:
    def __init__(self, points: List[Point]) -> None:
        self.points = points

def calculate_area(rectangle: Rectangle) -> float:
    height = rectangle.top_right.y - rectangle.bottom_left.y
    width = rectangle.top_right.x - rectangle.bottom_left.x
    return height * width

def calculate_center(rectangle: Rectangle) -> Point:
    center_x = (rectangle.bottom_left.x + rectangle.top_right.x) / 2
    center_y = (rectangle.bottom_left.y + rectangle.top_right.y) / 2
    return Point(center_x, center_y)

def calculate_distance(point1: Point, point2: Point) -> float:
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5

def calculate_circumference(circle: Circle) -> float:
    return 2 * pi * circle.radius

def calculate_circle_area(circle: Circle) -> float:
    return pi * (circle.radius ** 2)

def calculate_perimeter(polygon: Polygon) -> float:
    perimeter = 0
    points = polygon.points + [polygon.points[0]]  # Add the first point at the end for a closed shape
    for i in range(len(points) - 1):
        perimeter += calculate_distance(points[i], points[i + 1])
    return perimeter

# Create a rectangle
x = Rectangle(Point(2, 3), Point(6, 5))

# Create a circle
y = Circle(Point(0, 0), 5)

# Create a polygon
z = Polygon([Point(0, 0), Point(1, 0), Point(0, 1)])

# Calculate circumference
print(calculate_circumference("""
# %%
CODE2 = """
from typing import List
from math import pi

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

class A:
    def __init__(self, bottom_left: Point, top_right: Point) -> None:
        self.bottom_left = bottom_left
        self.top_right = top_right
        
class B:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

class C:
    def __init__(self, points: List[Point]) -> None:
        self.points = points

def calculate_area(rectangle: A) -> float:
    height = rectangle.top_right.y - rectangle.bottom_left.y
    width = rectangle.top_right.x - rectangle.bottom_left.x
    return height * width

def calculate_center(rectangle: A) -> Point:
    center_x = (rectangle.bottom_left.x + rectangle.top_right.x) / 2
    center_y = (rectangle.bottom_left.y + rectangle.top_right.y) / 2
    return Point(center_x, center_y)

def calculate_distance(point1: Point, point2: Point) -> float:
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5

def calculate_circumference(circle: B) -> float:
    return 2 * pi * circle.radius

def calculate_circle_area(circle: B) -> float:
    return pi * (circle.radius ** 2)

def calculate_perimeter(polygon: C) -> float:
    perimeter = 0
    points = polygon.points + [polygon.points[0]]  # Add the first point at the end for a closed shape
    for i in range(len(points) - 1):
        perimeter += calculate_distance(points[i], points[i + 1])
    return perimeter

foo = A(Point(2, 3), Point(6, 5))

bar = B(Point(0, 0), 5)

name = C([Point(0, 0), Point(1, 0), Point(0, 1)])

# Calculate circumference
print(calculate_circumference("""

# %%
shorter_prompt = """
class A:
    def __init__(self, bottom_left: Point, top_right: Point) -> None:
        self.bottom_left = bottom_left
        self.top_right = top_right
        
class B:
    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius

class C:
    def __init__(self, points: List[Point]) -> None:
        self.points = points

def calculate_circumference(circle: B) -> float:
    return 2 * pi * circle.radius

def calculate_area(rectangle: A) -> float:
    height = rectangle.top_right.y - rectangle.bottom_left.y
    width = rectangle.top_right.x - rectangle.bottom_left.x
    return height * width

def calculate_distance(point1: Point, point2: Point) -> float:
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2) ** 0.5

foo = A(Point(2, 3), Point(6, 5))

bar = B(Point(0, 0), 5)

name = C([Point(0, 0), Point(1, 0), Point(0, 1)])

# Calculate circumference
print(calculate_circumference("""

# %%
transformer_lens.utils.test_prompt(shorter_prompt, 'bar', pythia)
# %%
logits = pythia(CODE2)
predictions = torch.topk(logits[0,-1], 20)
pythia.to_str_tokens(predictions.indices)
# %%
threshold = 0.5
c = core.connectom(pythia, CODE2,
        core.logit_diff_metric(pythia, 'bar', 'foo','name'),
        core.ZeroPattern(),
        # core.BacktrackingStrategy(threshold),
        core.SplitStrategy(pythia, CODE2, threshold),
        max_batch_size=10
)
# %%
graph = core.plot_graphviz_connectome(pythia, CODE2, c, threshold=0.3).pipe('svg').decode('utf-8')
SVG(graph)
# %%
core.plot_graphviz_connectome(pythia, CODE2, c, threshold=threshold).render("img.svg")
# %%
core.plot_attn_connectome(pythia, CODE, c).show()

# %%
