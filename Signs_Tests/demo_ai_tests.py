#!/usr/bin/env python3
"""
AI Test Suite Demo Script for CMPE 187
=====================================

Runs all 27 AI test cases and displays for each:
- Input: Shows the input image first
- Expected Output: Expected number of signs/stoplights + expected bounding boxes (blue)
- Actual Output: Detected count + detected bounding boxes (red)
- Result: PASS or FAIL (count validation + localization IoU score)

Usage:
    python3 demo_ai_tests.py
"""

import os
import sys
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from google.cloud import vision
from tests.ai_tests.test_utils import (
    draw_bounding_boxes,
    find_image_path,
    filter_signs,
    create_json_output,
    save_json_output,
    setup_output_directory
)


def load_expected_localizations():
    """Load expected bounding box localizations from JSON file"""
    json_path = os.path.join(script_dir, 'expected_localizations.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format: {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}
    Returns IoU score between 0 and 1.
    """
    # Calculate intersection
    x_left = max(box1['x_min'], box2['x_min'])
    y_top = max(box1['y_min'], box2['y_min'])
    x_right = min(box1['x_max'], box2['x_max'])
    y_bottom = min(box1['y_max'], box2['y_max'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def convert_detected_to_box(detected_obj):
    """Convert a detected object's bounding poly to our box format"""
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
    """
    Calculate localization accuracy using IoU matching.
    Returns: (matched_count, avg_iou, details)
    """
    if not expected_boxes or not detected_objects:
        if not expected_boxes and not detected_objects:
            return 0, 1.0, []  # Both empty = perfect match
        return 0, 0.0, []

    # Convert detected objects to box format
    detected_boxes = [convert_detected_to_box(obj) for obj in detected_objects]

    # Greedy matching: for each expected box, find best matching detected box
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


# Terminal colors
class Colors:
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'


# Test case definitions
TEST_CASES = {
    # Base/Clear Scenarios
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
    # Augmented Scenarios
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


def show_input_image(image_path, test_id, description, expected_count):
    """Display the clean input image before processing (no bounding boxes)"""
    img = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(18, 10))  # Match input+output figure size
    ax.imshow(img)

    ax.set_title(f"Input: {test_id}\n{description}\nExpected: {expected_count} signs/stoplights", fontsize=10, fontweight='bold')
    ax.axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()


def show_result_image(output_image_path, test_id, expected, detected, passed):
    """Display the result image with bounding boxes"""
    img = mpimg.imread(output_image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"
    ax.set_title(f"RESULT: {test_id} - {status}\nExpected: {expected} | Detected: {detected}",
                 fontsize=14, fontweight='bold', color=color)
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # Show for 3 seconds
    plt.close()


def show_before_after(input_image_path, output_image_path, test_id, description, expected, detected, count_passed, loc_info=None, expected_boxes=None):
    """Display input (with expected blue boxes) and output (with detected red boxes) images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Input image with EXPECTED boxes in blue (left)
    img_input = Image.open(input_image_path)
    width, height = img_input.size
    axes[0].imshow(img_input)

    # Draw expected boxes on the INPUT image (left side) - matching style
    if expected_boxes:
        for i, box_data in enumerate(expected_boxes):
            box = box_data['bounding_box']
            # Convert normalized coordinates to pixel coordinates for polygon
            coords = [
                (box['x_min'] * width, box['y_min'] * height),
                (box['x_max'] * width, box['y_min'] * height),
                (box['x_max'] * width, box['y_max'] * height),
                (box['x_min'] * width, box['y_max'] * height)
            ]

            poly = patches.Polygon(coords, fill=False, edgecolor='blue', linewidth=3)
            axes[0].add_patch(poly)

            # Label with blue background box (matching style)
            min_x = box['x_min'] * width
            min_y = box['y_min'] * height
            sign_type = box_data.get('type', 'Sign/Stoplight')
            label = f"Expected {i+1}: {sign_type}"
            axes[0].text(min_x, min_y - 10, label,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7),
                        fontsize=10, color='white', weight='bold')

    # Input title: Test ID, description (smaller), Expected count
    input_title = f"Input: {test_id}\n{description}\nExpected: {expected} signs/stoplights"
    axes[0].set_title(input_title, fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Output image with DETECTED boxes (red) only (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)

    # Build status string
    count_status = "PASS" if count_passed else "FAIL"

    if loc_info:
        loc_status = f"IoU: {loc_info['avg_iou']*100:.1f}% ({loc_info['matched']}/{loc_info['total']} matched)"
        title = f"Output: Detected: {detected} signs/stoplights | Count: {count_status}\n{loc_status}"
    else:
        title = f"Output: Detected: {detected} signs/stoplights\n{count_status}"

    color = "green" if count_passed else "red"
    axes[1].set_title(title, fontsize=10, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.94, "Blue=Expected, Red=Detected", ha='center', fontsize=10)
    plt.subplots_adjust(top=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show(block=False)
    plt.pause(4)  # Show for 4 seconds
    plt.close()


def run_test(test_id, client, images_dir, output_dir, expected_localizations=None):
    """Execute a single test and return results including localization verification"""
    test_info = TEST_CASES.get(test_id, {'expected': 0, 'description': 'Unknown'})
    expected_count = test_info['expected']
    description = test_info['description']

    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Find image
    image_path = find_image_path(images_dir, test_id)
    if not image_path:
        return None

    # Get expected bounding boxes for this test
    expected_boxes = None
    if expected_localizations and 'test_cases' in expected_localizations:
        test_data = expected_localizations['test_cases'].get(test_id)
        if test_data and test_data.get('localizations'):
            expected_boxes = test_data['localizations']

    # Show input image FIRST before processing (clean, no boxes)
    print(f"\n  Showing input image...")
    show_input_image(image_path, test_id, description, expected_count)

    # Call Vision API
    print(f"  Calling Google Vision API...")
    with open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    all_objects = response.localized_object_annotations

    # Filter for signs/stoplights
    signs_detected = filter_signs(all_objects)
    detected_count = len(signs_detected)

    # Determine count pass/fail
    count_passed = detected_count == expected_count

    # Calculate localization score if expected boxes are available
    loc_info = None
    if expected_boxes is not None:
        matched_count, avg_iou, loc_details = calculate_localization_score(expected_boxes, signs_detected)
        loc_info = {
            'matched': matched_count,
            'total': len(expected_boxes) if expected_boxes else 0,
            'avg_iou': avg_iou,
            'details': loc_details
        }

    # Generate output image with ONLY detected (red) boxes - using original function
    output_image_path = os.path.join(output_dir, f'{test_id}_result.jpg')
    draw_bounding_boxes(image_path, signs_detected, output_image_path)

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
    json_data = create_json_output(test_id, signs_detected, output_files, timing_info, expected_count, count_passed)

    # Add localization info to JSON
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
        'description': description,
        'expected': expected_count,
        'detected': detected_count,
        'count_passed': count_passed,
        'loc_info': loc_info,
        'expected_boxes': expected_boxes,
        'input_image': image_path,
        'output_image': output_image_path,
        'sign_details': [(p.name, p.score) for p in signs_detected],
        'all_objects': [(o.name, o.score) for o in all_objects],
        'duration': duration
    }


def main():
    base_path = script_dir
    images_dir = os.path.join(base_path, 'images', 'ai_tests')
    output_dir = setup_output_directory(base_path)

    print()
    print("=" * 80)
    print("CMPE 187 - AI SIGNS & STOPLIGHTS DETECTION TEST SUITE DEMO")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(TEST_CASES)}")
    print(f"Output Directory: {output_dir}")
    print()

    # Load expected localizations
    print("Loading expected bounding box localizations...")
    expected_localizations = load_expected_localizations()
    if expected_localizations:
        print(f"  Loaded localizations for {len(expected_localizations.get('test_cases', {}))} test cases")
    else:
        print("  Warning: No expected localizations found - skipping IoU verification")
    print()

    # Initialize API client
    print("Initializing Google Cloud Vision API...")
    client = vision.ImageAnnotatorClient()
    print("API ready.\n")

    results = []
    test_ids = sorted(TEST_CASES.keys())

    # Start overall timer
    overall_start_time = time.time()

    for i, test_id in enumerate(test_ids, 1):
        test_info = TEST_CASES[test_id]

        print("=" * 80)
        print(f"TEST {i}/{len(test_ids)}: {test_id}")
        print("=" * 80)

        # INPUT
        print(f"\n{Colors.CYAN}[INPUT]{Colors.END}")
        print(f"  Test ID:     {test_id}")
        print(f"  Description: {test_info['description']}")

        # EXPECTED OUTPUT
        print(f"\n{Colors.CYAN}[EXPECTED OUTPUT]{Colors.END}")
        print(f"  Expected Signs/Stoplights Count: {test_info['expected']}")

        # Show expected bounding boxes info
        if expected_localizations and 'test_cases' in expected_localizations:
            test_data = expected_localizations['test_cases'].get(test_id)
            if test_data and test_data.get('localizations'):
                print(f"  Expected Bounding Boxes: {len(test_data['localizations'])}")
            else:
                print(f"  Expected Bounding Boxes: N/A")

        # Run test (this shows input image first, then processes)
        result = run_test(test_id, client, images_dir, output_dir, expected_localizations)

        if result is None:
            print(f"\n{Colors.RED}ERROR: Image not found for {test_id}{Colors.END}")
            continue

        # ACTUAL OUTPUT
        print(f"\n{Colors.CYAN}[ACTUAL OUTPUT]{Colors.END}")
        print(f"  Detected Signs/Stoplights Count: {result['detected']}")
        print(f"  All Objects Detected:  {len(result['all_objects'])}")

        if result['all_objects']:
            print(f"\n  Objects from API:")
            for name, score in result['all_objects']:
                name_lower = name.lower()
                is_sign = any(kw in name_lower for kw in ['sign', 'stop', 'traffic light', 'stoplight', 'traffic signal'])
                marker = " <-- SIGN/STOPLIGHT" if is_sign else ""
                print(f"    - {name}: {score*100:.1f}%{marker}")

        if result['sign_details']:
            print(f"\n  Signs/Stoplights detected:")
            for j, (name, score) in enumerate(result['sign_details'], 1):
                print(f"    Sign/Stoplight {j} ({name}): {score*100:.1f}% confidence")

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
                          result['loc_info'], result.get('expected_boxes'))

        results.append(result)
        print()

    # Calculate overall execution time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    # Summary
    print("=" * 80)
    print("TEST EXECUTION COMPLETE - SUMMARY")
    print("=" * 80)

    count_passed = sum(1 for r in results if r['count_passed'])
    count_failed = len(results) - count_passed
    count_pass_rate = (count_passed / len(results) * 100) if results else 0

    # Calculate localization statistics
    loc_results = [r for r in results if r.get('loc_info')]
    if loc_results:
        avg_iou_overall = sum(r['loc_info']['avg_iou'] for r in loc_results) / len(loc_results)
        total_expected_boxes = sum(r['loc_info']['total'] for r in loc_results)
        total_matched_boxes = sum(r['loc_info']['matched'] for r in loc_results)
        loc_good = sum(1 for r in loc_results if r['loc_info']['avg_iou'] >= 0.5)
        loc_acceptable = sum(1 for r in loc_results if 0.3 <= r['loc_info']['avg_iou'] < 0.5)
        loc_poor = sum(1 for r in loc_results if r['loc_info']['avg_iou'] < 0.3)

    print(f"\n{'Test ID':<10} {'Expected':>10} {'Detected':>10} {'Count':>10} {'IoU':>10}")
    print("-" * 55)

    for r in results:
        count_status = f"{Colors.GREEN}PASS{Colors.END}" if r['count_passed'] else f"{Colors.RED}FAIL{Colors.END}"
        if r.get('loc_info'):
            iou_pct = f"{r['loc_info']['avg_iou']*100:.0f}%"
        else:
            iou_pct = "N/A"
        print(f"{r['test_id']:<10} {r['expected']:>10} {r['detected']:>10} {count_status:>20} {iou_pct:>10}")

    print("-" * 55)

    # Count Test Summary
    print(f"\n{Colors.CYAN}COUNT VERIFICATION:{Colors.END}")
    print(f"  Total:     {len(results)} tests")
    print(f"  Passed:    {Colors.GREEN}{count_passed}{Colors.END}")
    print(f"  Failed:    {Colors.RED}{count_failed}{Colors.END}")
    print(f"  Pass Rate: {count_pass_rate:.1f}%")

    # Localization Summary
    if loc_results:
        print(f"\n{Colors.CYAN}LOCALIZATION VERIFICATION:{Colors.END}")
        print(f"  Tests with expected boxes: {len(loc_results)}")
        print(f"  Total expected boxes:      {total_expected_boxes}")
        print(f"  Total matched boxes:       {total_matched_boxes}")
        print(f"  Overall Avg IoU:           {avg_iou_overall*100:.1f}%")
        print(f"  Good (IoU >= 50%):         {Colors.GREEN}{loc_good}{Colors.END}")
        print(f"  Acceptable (IoU 30-50%):   {Colors.YELLOW}{loc_acceptable}{Colors.END}")
        print(f"  Poor (IoU < 30%):          {Colors.RED}{loc_poor}{Colors.END}")

    # Display overall execution time
    minutes = int(overall_duration // 60)
    seconds = overall_duration % 60
    if minutes > 0:
        print(f"\nTotal Execution Time: {minutes}m {seconds:.2f}s")
    else:
        print(f"\nTotal Execution Time: {seconds:.2f}s")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Images show: Blue = Expected, Red = Detected")
    print("=" * 80)


if __name__ == '__main__':
    main()
