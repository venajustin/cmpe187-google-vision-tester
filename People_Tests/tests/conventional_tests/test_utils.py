"""
Shared utilities for People Detection Tests

This module contains common functions used across all test cases including:
- Bounding box drawing
- JSON output generation
- Test result logging
- Image processing utilities
- Test timing and duration tracking
"""

import os
import json
import time
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
    # Load the image
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Get image dimensions
    img_height, img_width = img.shape[:2]

    # Draw bounding boxes only for people/pedestrians
    for i, obj in enumerate(people_objects, 1):
        # Get normalized vertices
        vertices = obj.bounding_poly.normalized_vertices

        # Convert normalized coordinates to pixel coordinates
        coords = [(vertex.x * img_width, vertex.y * img_height) for vertex in vertices]

        # Draw red bounding box for people
        poly = patches.Polygon(coords, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(poly)

        # Add label with object name, person number, and confidence
        if coords:
            label = f"Person {i}: {obj.name} ({obj.score:.2f})"
            # Find the top-left corner for label placement
            min_y = min(coord[1] for coord in coords)
            min_x = min(coord[0] for coord in coords)
            ax.text(min_x, min_y - 10, label,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                   fontsize=10, color='white', weight='bold')

    # Remove axes
    ax.axis('off')
    plt.tight_layout()

    # Save the image
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return output_path


def find_image_path(base_path, test_id):
    """
    Find the image file with common extensions.

    Args:
        base_path: Base directory path to search
        test_id: Test case ID (e.g., 'BVA-001')

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


def get_group_size_category(people_count):
    """
    Determine the group size category based on people count.

    Categories:
    - individual: 0-1 people
    - small: 2-5 people
    - medium: 6-10 people
    - large: 11-20 people
    - crowd: >20 people

    Args:
        people_count: Number of people (int)

    Returns:
        str: Group size category
    """
    if people_count <= 1:
        return 'individual'
    elif 2 <= people_count <= 5:
        return 'small'
    elif 6 <= people_count <= 10:
        return 'medium'
    elif 11 <= people_count <= 20:
        return 'large'
    else:  # > 20
        return 'crowd'


def calculate_metrics(actual_people, detected_people, group_size_category='small'):
    """
    Calculate detection metrics based on actual and detected counts.

    Args:
        actual_people: Ground truth number of people
        detected_people: Number of people detected by API
        group_size_category: 'small' (1-5), 'medium' (6-10), 'large' (11-20)

    Returns:
        dict: Dictionary containing calculated metrics
    """
    # Detection rate
    detection_rate = (detected_people / actual_people) * 100 if actual_people > 0 else 0

    # Count tolerance based on group size
    tolerance_map = {
        'small': 1,    # 1-5 people: ±1
        'medium': 2,   # 6-10 people: ±2
        'large': int(actual_people * 0.2)  # 11-20 people: ±20%
    }
    count_tolerance = tolerance_map.get(group_size_category, 1)

    # Count error
    count_error = abs(detected_people - actual_people)
    count_within_tolerance = count_error <= count_tolerance

    # False positives/negatives
    false_positives = max(0, detected_people - actual_people)
    false_negatives = max(0, actual_people - detected_people)

    return {
        'detection_rate': detection_rate,
        'count_tolerance': count_tolerance,
        'count_error': count_error,
        'count_within_tolerance': count_within_tolerance,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def format_duration(duration_seconds):
    """
    Format duration in seconds to a human-readable string.

    Args:
        duration_seconds: Duration in seconds (float)

    Returns:
        str: Formatted duration string
    """
    if duration_seconds < 1:
        return f"{duration_seconds * 1000:.2f}ms"
    elif duration_seconds < 60:
        return f"{duration_seconds:.2f}s"
    else:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        return f"{minutes}m {seconds:.2f}s"


def _standardize_expected_results(test_config, metrics):
    """
    Standardize the expected_results section to have consistent fields across all tests.

    Args:
        test_config: Dictionary containing test configuration
        metrics: Dictionary of calculated metrics

    Returns:
        dict: Standardized expected_results dictionary
    """
    # Build standardized expected_results with all fields
    expected_results = {
        "actual_people_in_scene": test_config['actual_people'],
        "detection_rate_threshold": test_config.get('detection_threshold', 85),
        "count_tolerance": metrics.get('count_tolerance', 0),
        "group_size_category": test_config.get('group_size_category', get_group_size_category(test_config['actual_people'])),
        "false_positives": 0,
        "false_negatives": 0
    }

    # Add confidence_threshold if specified in test config
    if 'confidence_threshold' in test_config:
        expected_results['confidence_threshold'] = test_config['confidence_threshold']

    # Add special notes for DT tests or other descriptive information
    # If test_config has 'expected_results' as a string (DT format), include it as 'test_expectations'
    if 'expected_results' in test_config and isinstance(test_config['expected_results'], str):
        expected_results['test_expectations'] = test_config['expected_results']

    # Add pass_criteria if specified
    if 'pass_criteria' in test_config:
        expected_results['pass_criteria'] = test_config['pass_criteria']

    return expected_results


def create_json_output(test_config, people_detected, metrics, test_passed, failure_reasons, output_files, timing_info=None):
    """
    Create standardized JSON output for test results.

    Args:
        test_config: Dictionary containing test configuration
        people_detected: List of detected people objects
        metrics: Dictionary of calculated metrics
        test_passed: Boolean indicating if test passed
        failure_reasons: List of failure reasons (empty if passed)
        output_files: Dictionary of output file paths
        timing_info: Optional dictionary with start_time, end_time, duration

    Returns:
        dict: JSON-serializable dictionary of test results
    """
    # Handle both BVA/EP format and DT format test configs
    json_result = {
        "test_case_id": test_config['test_id'],
        "test_name": test_config.get('test_name', test_config.get('description', 'Unknown')),
        "category": test_config['category'],
        "timestamp": datetime.now().isoformat(),
        "test_configuration": {
            "expected_people_count": test_config['actual_people'],
            "detection_threshold": test_config.get('detection_threshold', 85),
            "group_size_category": test_config.get('group_size_category', get_group_size_category(test_config['actual_people']))
        },
        "description": test_config.get('detailed_description', test_config.get('test_name', '')),
        "expected_results": _standardize_expected_results(test_config, metrics),
        "actual_results": {
            "actual_people_in_scene": test_config['actual_people'],
            "detected_people": len(people_detected),
            "detection_rate": round(metrics['detection_rate'], 2),
            "count_error": metrics['count_error'],
            "count_within_tolerance": metrics['count_within_tolerance'],
            "false_positives": metrics['false_positives'],
            "false_negatives": metrics['false_negatives'],
            "bounding_boxes_drawn": len(people_detected),
            "maximum_confidence": round(max(person.score for person in people_detected), 4) if people_detected else 0
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
        "test_result": {
            "status": "PASS" if test_passed else "FAIL",
            "meets_functional_requirements": test_passed,
            "failure_reasons": failure_reasons if not test_passed else []
        },
        "output_files": output_files
    }

    # Add optional fields if present in config
    if 'input_categories' in test_config:
        json_result['input_categories'] = test_config['input_categories']
    if 'comparison' in test_config:
        json_result['comparison'] = test_config['comparison']
    if 'test_reason' in test_config:
        json_result['test_result']['reason'] = test_config['test_reason']
    if 'pass_criteria' in test_config:
        json_result['pass_criteria'] = test_config['pass_criteria']

    # Add timing information if provided
    if timing_info:
        json_result["timing"] = {
            "start_time": timing_info['start_time'],
            "end_time": timing_info['end_time'],
            "duration_seconds": round(timing_info['duration'], 3),
            "duration_formatted": format_duration(timing_info['duration'])
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


def print_detection_summary(people_detected):
    """
    Print detection summary to console.

    Args:
        people_detected: List of detected people objects
    """
    print(f"\nDETECTION SUMMARY:")
    print(f"  - People/Pedestrians detected: {len(people_detected)}")

    if people_detected:
        print("\n  Detected people:")
        for i, person in enumerate(people_detected, 1):
            vertices = person.bounding_poly.normalized_vertices
            print(f"    Person {i}: {person.name} (confidence: {person.score:.4f})")
            print(f"      Bounding box: ({vertices[0].x:.3f}, {vertices[0].y:.3f}) to ({vertices[2].x:.3f}, {vertices[2].y:.3f})")


def setup_output_directory(base_path, subfolder=None):
    """
    Create output directory if it doesn't exist.

    Args:
        base_path: Base path for the test (usually project root)
        subfolder: Optional subfolder within results (e.g., 'conventional_tests', 'ai_tests')

    Returns:
        str: Path to output directory
    """
    # If base_path already ends with 'People_Tests', use it directly
    # Otherwise, assume it's the parent and go up one level
    if base_path.endswith('People_Tests'):
        output_dir = os.path.join(base_path, 'results')
    else:
        output_dir = os.path.join(
            os.path.dirname(base_path),
            'results'
        )

    # Add subfolder if specified
    if subfolder:
        output_dir = os.path.join(output_dir, subfolder)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class TeeOutput:
    """
    Class to redirect output to both console and file simultaneously.
    """
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, data):
        self.terminal.write(data)
        if self.file and not self.file.closed:
            try:
                self.file.write(data)
                self.file.flush()
            except ValueError:
                pass  # File already closed

    def flush(self):
        self.terminal.flush()
        if self.file and not self.file.closed:
            try:
                self.file.flush()
            except ValueError:
                pass  # File already closed
