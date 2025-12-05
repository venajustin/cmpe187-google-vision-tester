#!/usr/bin/env python3
"""
Unified AI Test Suite Demo Script for CMPE 187
===============================================

Runs ALL test cases from all three test suites:
- People Detection (28 tests): P-01, P-02, PT-03 through PT-28
- Vehicle Detection (19 tests): c-01 through c-19
- Signs & Stoplights Detection (27 tests): S-01 through S-27

Total: 74 test cases

For each test displays:
- Input: Shows the input image first
- Expected Output: Expected count + expected bounding boxes (blue)
- Actual Output: Detected count + detected bounding boxes (red)
- Result: PASS or FAIL (count validation + localization IoU score)

Usage:
    python3 demo_all_tests.py              # Run all 74 tests
    python3 demo_all_tests.py --people     # Run only people tests (28)
    python3 demo_all_tests.py --vehicles   # Run only vehicle tests (19)
    python3 demo_all_tests.py --signs      # Run only signs tests (27)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image

# Get script directory (project root)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add paths for imports from each test suite
sys.path.insert(0, os.path.join(script_dir, 'People_Tests'))
sys.path.insert(0, os.path.join(script_dir, 'Car_Tests'))
sys.path.insert(0, os.path.join(script_dir, 'Signs_Tests'))

from google.cloud import vision

# Import filter functions from each test suite
from People_Tests.tests.ai_tests.test_utils import filter_people
from Car_Tests.tests.ai_tests.test_utils import filter_vehicles
from Signs_Tests.tests.ai_tests.test_utils import filter_signs


# Terminal colors
class Colors:
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    END = '\033[0m'


# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================

PEOPLE_TEST_CASES = {
    'P-01':  {'expected': 1,  'description': 'Daylight, Clear, Quiet Street, Summer, 1 person, <10m'},
    'P-02':  {'expected': 4,  'description': 'Night, Clear, Busy Cityscape, Summer, 4 people, <10m'},
    'PT-03': {'expected': 1,  'description': 'Daylight, Rain, Quiet Street, Fall, 1 person, <10m'},
    'PT-04': {'expected': 3,  'description': 'Daylight, Snow, Busy Cityscape, Winter, 3 people, <10m'},
    'PT-05': {'expected': 0,  'description': 'Daylight, Clear, Busy Cityscape, Summer, 0 people'},
    'PT-06': {'expected': 1,  'description': 'Daylight, Clear, Busy Cityscape, Summer, 1 person, 10-50m'},
    'PT-07': {'expected': 2,  'description': 'Daylight, Clear, Quiet Street, Spring, 2 people, 10-50m'},
    'PT-08': {'expected': 1,  'description': 'Daylight, Fog, Quiet Street, Winter, 1 person, <10m'},
    'PT-09': {'expected': 6,  'description': 'Daylight, Clear, Busy Cityscape, Summer, 6 people, 10-50m'},
    'PT-10': {'expected': 12, 'description': 'Daylight, Clear, Busy Cityscape, Summer, 12 people, <10m'},
    'PT-11': {'expected': 21, 'description': 'Daylight, Clear, Busy Cityscape, Winter, 21 people, 10-50m'},
    'PT-12': {'expected': 4,  'description': 'Daylight, Clear, Busy Cityscape, Summer, 4 people, >50m'},
    'PT-13': {'expected': 1,  'description': 'Night, Snow, Busy Cityscape, Winter, 1 person, <10m'},
    'PT-14': {'expected': 1,  'description': 'Night, Rain, Busy Cityscape, Fall, 1 person, <10m'},
    'PT-15': {'expected': 1,  'description': 'Augmented P-01: Sepia | Daylight, Clear, Quiet Street, Summer, 1 person, <10m'},
    'PT-16': {'expected': 4,  'description': 'Augmented P-02: Blur | Night, Clear, Busy Cityscape, Summer, 4 people, <10m'},
    'PT-17': {'expected': 1,  'description': 'Augmented PT-03: Emboss | Daylight, Rain, Quiet Street, Fall, 1 person, <10m'},
    'PT-18': {'expected': 3,  'description': 'Augmented PT-04: Grayscale | Daylight, Snow, Busy Cityscape, Winter, 3 people, <10m'},
    'PT-19': {'expected': 0,  'description': 'Augmented PT-05: Sharpen | Daylight, Clear, Busy Cityscape, Summer, 0 people'},
    'PT-20': {'expected': 1,  'description': 'Augmented PT-06: Invert | Daylight, Clear, Busy Cityscape, Summer, 1 person, 10-50m'},
    'PT-21': {'expected': 2,  'description': 'Augmented PT-07: Sepia2 | Daylight, Clear, Quiet Street, Spring, 2 people, 10-50m'},
    'PT-22': {'expected': 1,  'description': 'Augmented PT-08: Brightness | Daylight, Fog, Quiet Street, Winter, 1 person, <10m'},
    'PT-23': {'expected': 6,  'description': 'Augmented PT-09: Remove White | Daylight, Clear, Busy Cityscape, Summer, 6 people, 10-50m'},
    'PT-24': {'expected': 12, 'description': 'Augmented PT-10: Noise | Daylight, Clear, Busy Cityscape, Summer, 12 people, <10m'},
    'PT-25': {'expected': 21, 'description': 'Augmented PT-11: Pixelate | Daylight, Clear, Busy Cityscape, Winter, 21 people, 10-50m'},
    'PT-26': {'expected': 4,  'description': 'Augmented PT-12: Color Filter | Daylight, Clear, Busy Cityscape, Summer, 4 people, >50m'},
    'PT-27': {'expected': 1,  'description': 'Augmented PT-13: Invert, Noise, Color Filter | Night, Snow, Busy Cityscape, Winter, 1 person, <10m'},
    'PT-28': {'expected': 1,  'description': 'Augmented PT-14: Grayscale, Emboss, Sharpen, Pixelate | Night, Rain, Busy Cityscape, Fall, 1 person, <10m'},
}

VEHICLE_TEST_CASES = {
    'c-01':  {'expected': 6,  'description': '1 to 5 cars, far distance, overlapping, car type: vehicles'},
    'c-02':  {'expected': 6,  'description': 'Augmented C-01: Heavy Pixelation, 1 to 5 cars, far distance, overlapping, car type: vehicles'},
    'c-03': {'expected': 2,  'description': '2 cars, medium distance, overlapping'},
    'c-04': {'expected': 2,  'description': 'Augmented C-03: Sepia2, 2 cars, medium distance, overlapping'},
    'c-05': {'expected': 1,  'description': '1 truck, medium distance, fall, clear weather, bright lighting, no overlapping'},
    'c-06': {'expected': 1,  'description': '1 truck, 1 car, medium distance, fall, dim lighting, clear weather, no overlapping'},
    'c-07': {'expected': 1,  'description': '1 truck, close distance, fall, dim lighting, rainy weather, no overlapping'},
    'c-08': {'expected': 2,  'description': '2 trucks, far distance, winter, bright lighting, snowy weather, no overlapping'},
    'c-09': {'expected': 3,  'description': '3 cars and 1 truck, snowy weather, dim lighting, overlapping, medium distance'},
    'c-10': {'expected': 3, 'description': '3 cars, rainy weather, winter season, close distance'},
    'c-11': {'expected': 4, 'description': '4 cars, close distance, rainy weather, fall season, overlapping'},
    'c-12': {'expected': 1,  'description': '1 car, winter season, dim lighting, clear weather'},
    'c-13': {'expected': 2,  'description': '2 cars, fall season, blurry image, medium distance'},
    'c-14': {'expected': 2,  'description': 'Augmented C13: monochromatic (black/white), 2 cars, fall season, blurry image, medium distance'},
    'c-15': {'expected': 14,  'description': '5+ cars, far distance, overlapping, spring season, clear weather, bright lighting'},
    'c-16': {'expected': 6,  'description': '6 cars, overlapping, far distance, dim lighting, summer season, clear weather'},
    'c-17': {'expected': 1,  'description': '1 truck, far distance, bright lighting, summer season, clear weather'},
    'c-18': {'expected': 0,  'description': 'No cars, dim lighting, fall, clear weather'},
    'c-19': {'expected': 0,  'description': 'Augmentation C-18: Blur, No cars, dim lighting, summer weather'},
}

SIGNS_TEST_CASES = {
    'S-01': {'expected': 2, 'description': 'A stoplight and a regulatory sign on the street in foggy weather during winter in dim lighting.'},
    'S-02': {'expected': 10, 'description': 'An intersection with 8 stoplights, 1 regulatory sign, and 1 street sign in clear weather during spring in normal lighting.'},
    'S-03': {'expected': 10, 'description': 'An intersection with 8 stoplights, 1 regulatory sign, and 1 street sign in clear weather during winter in normal lighting.'},
    'S-04': {'expected': 0, 'description': 'A street with no stoplights or signs in foggy weather during winter in normal lighting.'},
    'S-05': {'expected': 0, 'description': 'A street with no stoplights or signs in clear weather during summer in normal lighting.'},
    'S-06': {'expected': 0, 'description': 'A street with no stoplights or signs in clear weather during fall in normal lighting.'},
    'S-07': {'expected': 1, 'description': 'A street with 1 stop sign in clear weather during spring in dim lighting.'},
    'S-08': {'expected': 1, 'description': 'A street with 1 regulatory sign in clear weather during summer in normal lighting.'},
    'S-09': {'expected': 1, 'description': 'A street with 1 warning sign in clear weather during summer in normal lighting.'},
    'S-10': {'expected': 1, 'description': 'A street with 1 street sign in clear weather during winter in normal lighting.'},
    'S-11': {'expected': 1, 'description': 'A street with 1 speed limit sign in clear weather during fall in normal lighting.'},
    'S-12': {'expected': 3, 'description': 'An image of 3 stoplights in clear weather during summer in normal lighting.'},
    'S-13': {'expected': 4, 'description': 'A street with 1 stop sign, 2 street signs, and 1 speed limit sign in clear weather during summer in normal lighting.'},
    'S-14': {'expected': 1, 'description': 'A street with 1 stop sign in rainy weather during fall in normal lighting.'},
    'S-15': {'expected': 10, 'description': 'Augmented S-02: Sepia at normal brightness.'},
    'S-16': {'expected': 10, 'description': 'Augmented S-03: Sharpen at low brightness.'},
    'S-17': {'expected': 0, 'description': 'Augmented S-04: Low noise with bright lighting.'},
    'S-18': {'expected': 0, 'description': 'Augmented S-05: Emboss at normal brightness.'},
    'S-19': {'expected': 0, 'description': 'Augmented S-06: Low blurriness at normal brightness.'},
    'S-20': {'expected': 1, 'description': 'Augmented S-07: Low brightness.'},
    'S-21': {'expected': 1, 'description': 'Augmented S-08: Grayscale at normal brightness.'},
    'S-22': {'expected': 1, 'description': 'Augmented S-09: High noise.'},
    'S-23': {'expected': 1, 'description': 'Augmented S-10: High pixelation'},
    'S-24': {'expected': 1, 'description': 'Augmented S-11: Emboss and high brightness.'},
    'S-25': {'expected': 3, 'description': 'Augmented S-12: Invert at normal brightness.'},
    'S-26': {'expected': 4, 'description': 'Augmented S-13: Grayscale with low pixelation at normal brightness.'},
    'S-27': {'expected': 1, 'description': 'Augmented S-14: Blur at normal brightness.'}
}

# ============================================================================
# TEST SUITE CONFIGURATION
# ============================================================================

TEST_SUITES = {
    'people': {
        'name': 'People Detection',
        'base_dir': 'People_Tests',
        'filter_func': filter_people,
        'label': 'Person',
        'object_type': 'people',
        'marker_keywords': ['person', 'people', 'pedestrian'],
        'test_cases': PEOPLE_TEST_CASES
    },
    'vehicles': {
        'name': 'Vehicle Detection',
        'base_dir': 'Car_Tests',
        'filter_func': filter_vehicles,
        'label': 'Vehicle',
        'object_type': 'vehicles',
        'marker_keywords': ['car', 'truck', 'vehicle', 'bus'],
        'test_cases': VEHICLE_TEST_CASES
    },
    'signs': {
        'name': 'Signs & Stoplights Detection',
        'base_dir': 'Signs_Tests',
        'filter_func': filter_signs,
        'label': 'Sign/Stoplight',
        'object_type': 'signs/stoplights',
        'marker_keywords': ['sign', 'stop', 'traffic light', 'stoplight', 'traffic signal'],
        'test_cases': SIGNS_TEST_CASES
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_image_path(images_dir, test_id):
    """Find image file with any supported extension"""
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(images_dir, f"{test_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_expected_localizations(suite_dir):
    """Load expected bounding box localizations from JSON file"""
    json_path = os.path.join(suite_dir, 'expected_localizations.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x_left = max(box1['x_min'], box2['x_min'])
    y_top = max(box1['y_min'], box2['y_min'])
    x_right = min(box1['x_max'], box2['x_max'])
    y_bottom = min(box1['y_max'], box2['y_max'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def convert_detected_to_box(detected_obj):
    """Convert a detected object's bounding poly to box format"""
    vertices = detected_obj.bounding_poly.normalized_vertices
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return {
        'x_min': min(x_coords),
        'y_min': min(y_coords),
        'x_max': max(x_coords),
        'y_max': max(y_coords)
    }


def calculate_localization_score(expected_boxes, detected_objects, iou_threshold=0.3):
    """Calculate localization accuracy using IoU matching"""
    if not expected_boxes or not detected_objects:
        if not expected_boxes and not detected_objects:
            return 0, 1.0, []
        return 0, 0.0, []

    detected_boxes = [convert_detected_to_box(obj) for obj in detected_objects]
    matched_ious = []
    used_detected = set()
    details = []

    for i, exp_box in enumerate(expected_boxes):
        best_iou = 0.0
        best_idx = -1

        for j, det_box in enumerate(detected_boxes):
            if j in used_detected:
                continue
            iou = calculate_iou(exp_box['bounding_box'], det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0 and best_iou >= iou_threshold:
            used_detected.add(best_idx)
            matched_ious.append(best_iou)
            details.append({'expected_id': i+1, 'detected_id': best_idx+1, 'iou': best_iou, 'matched': True})
        else:
            details.append({'expected_id': i+1, 'detected_id': None, 'iou': best_iou, 'matched': False})

    matched_count = len(matched_ious)
    avg_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return matched_count, avg_iou, details


def setup_output_directory(suite_dir):
    """Create output directory for results"""
    output_dir = os.path.join(suite_dir, 'results', 'ai_tests')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def draw_bounding_boxes(image_path, detected_objects, output_path, label_prefix):
    """Draw bounding boxes on image with custom label prefix using matplotlib for better visibility"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    img_height, img_width = img.shape[:2]

    for i, obj in enumerate(detected_objects, 1):
        vertices = obj.bounding_poly.normalized_vertices
        coords = [(v.x * img_width, v.y * img_height) for v in vertices]

        # Draw polygon with thick red line
        poly = patches.Polygon(coords, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(poly)

        # Draw label with red background
        if coords:
            label = f"{label_prefix} {i}: {obj.name} ({obj.score:.2f})"
            min_y = min(coord[1] for coord in coords)
            min_x = min(coord[0] for coord in coords)
            ax.text(min_x, min_y - 10, label,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                   fontsize=10, color='white', weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


def create_json_output(test_id, detected_objects, output_files, timing_info, expected_count, passed):
    """Create JSON output data"""
    detected_details = []
    for i, obj in enumerate(detected_objects, 1):
        vertices = obj.bounding_poly.normalized_vertices
        detected_details.append({
            'sign_id': i,
            'name': obj.name,
            'confidence': round(obj.score, 4),
            'bounding_box': {
                'normalized_vertices': [
                    {'x': round(v.x, 4), 'y': round(v.y, 4)} for v in vertices
                ]
            }
        })

    return {
        'test_case_id': test_id,
        'category': 'AI Generated Test',
        'timestamp': datetime.now().isoformat(),
        'detection_results': {
            'expected_count': expected_count,
            'detected_count': len(detected_objects),
            'count_difference': len(detected_objects) - expected_count,
            'maximum_confidence': round(max((obj.score for obj in detected_objects), default=0), 4)
        },
        'test_result': {
            'status': 'PASS' if passed else 'FAIL',
            'passed': passed
        },
        'detected_sign_details': detected_details,
        'output_files': output_files,
        'timing': {
            'start_time': timing_info['start_time'],
            'end_time': timing_info['end_time'],
            'duration_seconds': round(timing_info['duration'], 3)
        }
    }


def save_json_output(json_data, output_path):
    """Save JSON data to file"""
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def show_input_image(image_path, test_id, description, expected_count, object_type):
    """Display the clean input image before processing"""
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img)
    ax.set_title(f"Input: {test_id}\n{description}\nExpected: {expected_count} {object_type}", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def show_before_after(input_image_path, output_image_path, test_id, description, expected, detected,
                      count_passed, object_type, label_prefix, loc_info=None, expected_boxes=None):
    """Display input and output images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Input image with EXPECTED boxes in blue (left)
    img_input = Image.open(input_image_path)
    width, height = img_input.size
    axes[0].imshow(img_input)

    if expected_boxes:
        for i, box_data in enumerate(expected_boxes):
            box = box_data['bounding_box']
            coords = [
                (box['x_min'] * width, box['y_min'] * height),
                (box['x_max'] * width, box['y_min'] * height),
                (box['x_max'] * width, box['y_max'] * height),
                (box['x_min'] * width, box['y_max'] * height)
            ]
            poly = patches.Polygon(coords, fill=False, edgecolor='blue', linewidth=3)
            axes[0].add_patch(poly)

            min_x = box['x_min'] * width
            min_y = box['y_min'] * height
            box_type = box_data.get('type', label_prefix)
            label = f"Expected {i+1}: {box_type}"
            axes[0].text(min_x, min_y - 10, label,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7),
                        fontsize=10, color='white', weight='bold')

    input_title = f"Input: {test_id}\n{description}\nExpected: {expected} {object_type}"
    axes[0].set_title(input_title, fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Output image with DETECTED boxes (red) (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)

    count_status = "PASS" if count_passed else "FAIL"
    if loc_info:
        loc_status = f"IoU: {loc_info['avg_iou']*100:.1f}% ({loc_info['matched']}/{loc_info['total']} matched)"
        title = f"Output: Detected: {detected} {object_type} | Count: {count_status}\n{loc_status}"
    else:
        title = f"Output: Detected: {detected} {object_type}\n{count_status}"

    color = "green" if count_passed else "red"
    axes[1].set_title(title, fontsize=10, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.94, "Blue=Expected, Red=Detected", ha='center', fontsize=10)
    plt.subplots_adjust(top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=False)
    plt.pause(4)
    plt.close()


# ============================================================================
# TEST EXECUTION
# ============================================================================

def run_test(test_id, client, suite_config, images_dir, output_dir, expected_localizations=None):
    """Execute a single test and return results"""
    test_cases = suite_config['test_cases']
    test_info = test_cases.get(test_id, {'expected': 0, 'description': 'Unknown'})
    expected_count = test_info['expected']
    description = test_info['description']

    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Find image
    image_path = find_image_path(images_dir, test_id)
    if not image_path:
        return None

    # Get expected bounding boxes
    expected_boxes = None
    if expected_localizations and 'test_cases' in expected_localizations:
        test_data = expected_localizations['test_cases'].get(test_id)
        if test_data and test_data.get('localizations'):
            expected_boxes = test_data['localizations']

    # Show input image
    print(f"\n  Showing input image...")
    show_input_image(image_path, test_id, description, expected_count, suite_config['object_type'])

    # Call Vision API
    print(f"  Calling Google Vision API...")
    with open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    all_objects = response.localized_object_annotations

    # Filter using suite-specific filter function
    filter_func = suite_config['filter_func']
    filtered_objects = filter_func(all_objects)
    detected_count = len(filtered_objects)

    # Determine count pass/fail
    count_passed = detected_count == expected_count

    # Calculate localization score
    loc_info = None
    if expected_boxes is not None:
        matched_count, avg_iou, loc_details = calculate_localization_score(expected_boxes, filtered_objects)
        loc_info = {
            'matched': matched_count,
            'total': len(expected_boxes) if expected_boxes else 0,
            'avg_iou': avg_iou,
            'details': loc_details
        }

    # Generate output image with detected boxes
    output_image_path = os.path.join(output_dir, f'{test_id}_result.jpg')
    draw_bounding_boxes(image_path, filtered_objects, output_image_path, suite_config['label'])

    # Save JSON
    end_time = time.time()
    duration = end_time - start_time

    timing_info = {
        'start_time': start_timestamp,
        'end_time': datetime.now().isoformat(),
        'duration': duration
    }
    output_files = {
        'result_image': output_image_path,
        'output_json': os.path.join(output_dir, f'{test_id}_output.json')
    }
    json_data = create_json_output(test_id, filtered_objects, output_files, timing_info, expected_count, count_passed)

    if loc_info:
        json_data['localization_results'] = {
            'expected_boxes': len(expected_boxes) if expected_boxes else 0,
            'matched_boxes': loc_info['matched'],
            'average_iou': round(loc_info['avg_iou'], 4),
            'iou_threshold': 0.3,
            'details': loc_info['details']
        }

    save_json_output(json_data, output_files['output_json'])

    return {
        'test_id': test_id,
        'suite': suite_config['name'],
        'description': description,
        'expected': expected_count,
        'detected': detected_count,
        'count_passed': count_passed,
        'loc_info': loc_info,
        'expected_boxes': expected_boxes,
        'input_image': image_path,
        'output_image': output_image_path,
        'object_details': [(p.name, p.score) for p in filtered_objects],
        'all_objects': [(o.name, o.score) for o in all_objects],
        'duration': duration
    }


def run_suite(suite_key, suite_config, client, overall_start_time):
    """Run all tests in a suite"""
    suite_dir = os.path.join(script_dir, suite_config['base_dir'])
    images_dir = os.path.join(suite_dir, 'images', 'ai_tests')
    output_dir = setup_output_directory(suite_dir)

    print()
    print("*" * 80)
    print(f"  {suite_config['name'].upper()} TEST SUITE")
    print("*" * 80)
    print(f"  Test Cases: {len(suite_config['test_cases'])}")
    print(f"  Images Dir: {images_dir}")
    print(f"  Output Dir: {output_dir}")

    # Load expected localizations
    print("\n  Loading expected bounding box localizations...")
    expected_localizations = load_expected_localizations(suite_dir)
    if expected_localizations:
        print(f"    Loaded localizations for {len(expected_localizations.get('test_cases', {}))} test cases")
    else:
        print("    Warning: No expected localizations found - skipping IoU verification")

    results = []
    test_ids = sorted(suite_config['test_cases'].keys())

    for i, test_id in enumerate(test_ids, 1):
        test_info = suite_config['test_cases'][test_id]

        print()
        print("=" * 80)
        print(f"TEST {i}/{len(test_ids)}: {test_id} ({suite_config['name']})")
        print("=" * 80)

        # INPUT
        print(f"\n{Colors.CYAN}[INPUT]{Colors.END}")
        print(f"  Test ID:     {test_id}")
        print(f"  Description: {test_info['description']}")

        # EXPECTED OUTPUT
        print(f"\n{Colors.CYAN}[EXPECTED OUTPUT]{Colors.END}")
        print(f"  Expected {suite_config['label']} Count: {test_info['expected']}")

        if expected_localizations and 'test_cases' in expected_localizations:
            test_data = expected_localizations['test_cases'].get(test_id)
            if test_data and test_data.get('localizations'):
                print(f"  Expected Bounding Boxes: {len(test_data['localizations'])}")
            else:
                print(f"  Expected Bounding Boxes: N/A")

        # Run test
        result = run_test(test_id, client, suite_config, images_dir, output_dir, expected_localizations)

        if result is None:
            print(f"\n{Colors.RED}ERROR: Image not found for {test_id}{Colors.END}")
            continue

        # ACTUAL OUTPUT
        print(f"\n{Colors.CYAN}[ACTUAL OUTPUT]{Colors.END}")
        print(f"  Detected {suite_config['label']} Count: {result['detected']}")
        print(f"  All Objects Detected: {len(result['all_objects'])}")

        if result['all_objects']:
            print(f"\n  Objects from API:")
            for name, score in result['all_objects']:
                name_lower = name.lower()
                is_target = any(kw in name_lower for kw in suite_config['marker_keywords'])
                marker = f" <-- {suite_config['label'].upper()}" if is_target else ""
                print(f"    - {name}: {score*100:.1f}%{marker}")

        if result['object_details']:
            print(f"\n  {suite_config['label']}s detected:")
            for j, (name, score) in enumerate(result['object_details'], 1):
                print(f"    {suite_config['label']} {j} ({name}): {score*100:.1f}% confidence")

        # LOCALIZATION RESULTS
        if result['loc_info']:
            print(f"\n{Colors.CYAN}[LOCALIZATION VERIFICATION]{Colors.END}")
            loc = result['loc_info']
            print(f"  Boxes Matched: {loc['matched']}/{loc['total']}")
            print(f"  Average IoU:   {loc['avg_iou']*100:.1f}%")
            if loc['avg_iou'] >= 0.5:
                print(f"  Localization:  {Colors.GREEN}GOOD{Colors.END} (IoU >= 50%)")
            elif loc['avg_iou'] >= 0.3:
                print(f"  Localization:  {Colors.YELLOW}ACCEPTABLE{Colors.END} (IoU >= 30%)")
            else:
                print(f"  Localization:  {Colors.RED}POOR{Colors.END} (IoU < 30%)")

        # RESULT
        print(f"\n{Colors.CYAN}[RESULT]{Colors.END}")
        if result['count_passed']:
            print(f"  Count Test: {Colors.GREEN}{Colors.BOLD}PASS{Colors.END} - Expected {result['expected']}, Detected {result['detected']}")
        else:
            print(f"  Count Test: {Colors.RED}{Colors.BOLD}FAIL{Colors.END} - Expected {result['expected']}, Detected {result['detected']}")

        print(f"  Duration: {result['duration']:.2f}s")

        # Show before/after comparison
        print(f"\n  Showing result comparison (Blue=Expected, Red=Detected)...")
        show_before_after(result['input_image'], result['output_image'],
                          test_id, result['description'],
                          result['expected'], result['detected'], result['count_passed'],
                          suite_config['object_type'], suite_config['label'],
                          result['loc_info'], result.get('expected_boxes'))

        results.append(result)

    return results


def print_suite_summary(suite_name, results):
    """Print summary for a single suite"""
    if not results:
        return

    count_passed = sum(1 for r in results if r['count_passed'])
    count_failed = len(results) - count_passed
    pass_rate = (count_passed / len(results) * 100) if results else 0

    print(f"\n  {Colors.MAGENTA}{suite_name}:{Colors.END}")
    print(f"    Tests: {len(results)} | Passed: {Colors.GREEN}{count_passed}{Colors.END} | Failed: {Colors.RED}{count_failed}{Colors.END} | Rate: {pass_rate:.1f}%")

    loc_results = [r for r in results if r.get('loc_info')]
    if loc_results:
        avg_iou = sum(r['loc_info']['avg_iou'] for r in loc_results) / len(loc_results)
        print(f"    Avg IoU: {avg_iou*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Run unified AI test suite demo')
    parser.add_argument('--people', action='store_true', help='Run only people detection tests')
    parser.add_argument('--vehicles', action='store_true', help='Run only vehicle detection tests')
    parser.add_argument('--signs', action='store_true', help='Run only signs detection tests')
    args = parser.parse_args()

    # Determine which suites to run
    suites_to_run = []
    if args.people or args.vehicles or args.signs:
        if args.people:
            suites_to_run.append('people')
        if args.vehicles:
            suites_to_run.append('vehicles')
        if args.signs:
            suites_to_run.append('signs')
    else:
        suites_to_run = ['people', 'vehicles', 'signs']

    total_tests = sum(len(TEST_SUITES[s]['test_cases']) for s in suites_to_run)

    print()
    print("=" * 80)
    print("CMPE 187 - UNIFIED AI DETECTION TEST SUITE DEMO")
    print("=" * 80)
    print(f"\nTest Suites: {', '.join(TEST_SUITES[s]['name'] for s in suites_to_run)}")
    print(f"Total Test Cases: {total_tests}")
    print()

    # Initialize API client
    print("Initializing Google Cloud Vision API...")
    client = vision.ImageAnnotatorClient()
    print("API ready.\n")

    # Run all selected suites
    overall_start_time = time.time()
    all_results = {}

    for suite_key in suites_to_run:
        suite_config = TEST_SUITES[suite_key]
        results = run_suite(suite_key, suite_config, client, overall_start_time)
        all_results[suite_key] = results

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    # ========================================================================
    # COMBINED SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("UNIFIED TEST EXECUTION COMPLETE - COMBINED SUMMARY")
    print("=" * 80)

    # Per-suite summaries
    for suite_key in suites_to_run:
        print_suite_summary(TEST_SUITES[suite_key]['name'], all_results.get(suite_key, []))

    # Overall statistics
    all_test_results = []
    for results in all_results.values():
        all_test_results.extend(results)

    total_passed = sum(1 for r in all_test_results if r['count_passed'])
    total_failed = len(all_test_results) - total_passed
    overall_pass_rate = (total_passed / len(all_test_results) * 100) if all_test_results else 0

    print(f"\n{Colors.CYAN}{'='*40}{Colors.END}")
    print(f"{Colors.CYAN}OVERALL RESULTS{Colors.END}")
    print(f"{Colors.CYAN}{'='*40}{Colors.END}")
    print(f"  Total Tests:   {len(all_test_results)}")
    print(f"  Total Passed:  {Colors.GREEN}{total_passed}{Colors.END}")
    print(f"  Total Failed:  {Colors.RED}{total_failed}{Colors.END}")
    print(f"  Pass Rate:     {overall_pass_rate:.1f}%")

    # Overall localization stats
    all_loc_results = [r for r in all_test_results if r.get('loc_info')]
    if all_loc_results:
        avg_iou_overall = sum(r['loc_info']['avg_iou'] for r in all_loc_results) / len(all_loc_results)
        total_expected = sum(r['loc_info']['total'] for r in all_loc_results)
        total_matched = sum(r['loc_info']['matched'] for r in all_loc_results)
        print(f"\n{Colors.CYAN}LOCALIZATION SUMMARY:{Colors.END}")
        print(f"  Tests with boxes: {len(all_loc_results)}")
        print(f"  Expected boxes:   {total_expected}")
        print(f"  Matched boxes:    {total_matched}")
        print(f"  Overall Avg IoU:  {avg_iou_overall*100:.1f}%")

    # Execution time
    minutes = int(overall_duration // 60)
    seconds = overall_duration % 60
    if minutes > 0:
        print(f"\nTotal Execution Time: {minutes}m {seconds:.2f}s")
    else:
        print(f"\nTotal Execution Time: {seconds:.2f}s")

    print("\nResults saved to each suite's results/ai_tests/ directory")
    print("  - Images show: Blue = Expected, Red = Detected")
    print("=" * 80)


if __name__ == '__main__':
    main()
