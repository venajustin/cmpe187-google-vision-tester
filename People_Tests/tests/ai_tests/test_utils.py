"""
Shared utilities for AI People Detection Tests

This module contains common functions used across all AI test cases including:
- Bounding box drawing
- JSON output generation
- Image processing utilities
"""

import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


def draw_bounding_boxes(image_path, people_objects, output_path):
    """
    Draw bounding boxes on the image for detected people/pedestrians only.

    Args:
        image_path: Path to the input image
        people_objects: List of detected people with bounding boxes
        output_path: Path to save the output image with bounding boxes

    Returns:
        str: Path to the saved output image
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    img_height, img_width = img.shape[:2]

    for i, obj in enumerate(people_objects, 1):
        vertices = obj.bounding_poly.normalized_vertices
        coords = [(vertex.x * img_width, vertex.y * img_height) for vertex in vertices]

        poly = patches.Polygon(coords, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(poly)

        if coords:
            label = f"Person {i}: {obj.name} ({obj.score:.2f})"
            min_y = min(coord[1] for coord in coords)
            min_x = min(coord[0] for coord in coords)
            ax.text(min_x, min_y - 10, label,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                   fontsize=10, color='white', weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return output_path


def find_image_path(base_path, test_id):
    """
    Find the image file with common extensions.

    Args:
        base_path: Base directory path to search
        test_id: Test case ID (e.g., 'PT-01')

    Returns:
        str: Full path to the image file, or None if not found
    """
    image_path = os.path.join(base_path, test_id)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    for ext in image_extensions:
        test_path = image_path + ext
        if os.path.exists(test_path):
            return test_path

    return None


def filter_people(objects):
    """
    Filter detected objects to only include people/pedestrians.

    Args:
        objects: List of all detected objects from Vision API

    Returns:
        list: Filtered list containing only people/pedestrian objects
    """
    return [obj for obj in objects if obj.name.lower() in ['person', 'people', 'pedestrian']]


def create_json_output(test_id, people_detected, output_files, timing_info=None, expected_count=None, test_passed=None):
    """
    Create simplified JSON output for AI test results.

    Args:
        test_id: Test case ID
        people_detected: List of detected people objects
        output_files: Dictionary of output file paths
        timing_info: Optional dictionary with start_time, end_time, duration
        expected_count: Expected number of people in the image
        test_passed: Boolean indicating if test passed

    Returns:
        dict: JSON-serializable dictionary of test results
    """
    detected_count = len(people_detected)

    json_result = {
        "test_case_id": test_id,
        "category": "AI Generated Test",
        "timestamp": datetime.now().isoformat(),
        "detection_results": {
            "expected_count": expected_count,
            "detected_count": detected_count,
            "count_difference": detected_count - expected_count if expected_count is not None else None,
            "maximum_confidence": round(max(person.score for person in people_detected), 4) if people_detected else 0
        },
        "test_result": {
            "status": "PASS" if test_passed else "FAIL",
            "passed": test_passed
        },
        "detected_people_details": [
            {
                "person_id": i,
                "name": person.name,
                "confidence": round(person.score, 4),
                "bounding_box": {
                    "normalized_vertices": [
                        {"x": round(v.x, 4), "y": round(v.y, 4)}
                        for v in person.bounding_poly.normalized_vertices
                    ]
                }
            }
            for i, person in enumerate(people_detected, 1)
        ],
        "output_files": output_files
    }

    if timing_info:
        json_result["timing"] = {
            "start_time": timing_info['start_time'],
            "end_time": timing_info['end_time'],
            "duration_seconds": round(timing_info['duration'], 3)
        }

    return json_result


def save_json_output(json_data, output_path):
    """
    Save JSON data to file.

    Args:
        json_data: Dictionary to save as JSON
        output_path: Path to save JSON file

    Returns:
        str: Path to saved JSON file
    """
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

    return output_path


def setup_output_directory(base_path):
    """
    Create output directory if it doesn't exist.

    Args:
        base_path: Base path (People_Tests directory)

    Returns:
        str: Path to output directory
    """
    output_dir = os.path.join(base_path, 'results', 'ai_tests')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
