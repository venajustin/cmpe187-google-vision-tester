#!/usr/bin/env python3
"""
AI Test Suite Demo Script for CMPE 187
=====================================

Runs all 27 AI test cases and displays for each:
- Input: Shows the input image first
- Expected Output: Expected number of signs and stoplights
- Actual Output: Detected count + annotated image with bounding boxes
- Result: PASS or FAIL

Usage:
    python3 demo_ai_tests.py
"""

from tests.ai_tests.test_utils import (
    # draw_bounding_boxes,
    find_image_path,
    filter_people,
    create_json_output,
    save_json_output,
    setup_output_directory
)
from google.cloud import vision
import os
import sys
import time
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


# Terminal colors
class Colors:
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'


def draw_boxes_signs(image_path, detected_objects, output_path):
    """
    Local function to draw bounding boxes with generic object labeling for signs/lights.
    (This is a modified copy of the original draw_bounding_boxes.)
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    img_height, img_width = img.shape[:2]

    # Use red for traffic objects
    box_color = 'red'

    for i, obj in enumerate(detected_objects, 1):
        vertices = obj.bounding_poly.normalized_vertices
        coords = [(vertex.x * img_width, vertex.y * img_height)
                  for vertex in vertices]

        poly = patches.Polygon(
            coords, fill=False, edgecolor=box_color, linewidth=3)
        ax.add_patch(poly)

        if coords:
            # *** MODIFIED LABELING LOGIC HERE ***
            # Display: [Object Name] # [Number] ([Confidence]%)
            label = f"{obj.name} #{i} ({obj.score:.2f})"

            min_y = min(coord[1] for coord in coords)
            min_x = min(coord[0] for coord in coords)
            ax.text(min_x, min_y - 10, label,
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor=box_color, alpha=0.7),
                    fontsize=10, color='white', weight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    return output_path


# Test case definitions
TEST_CASES_S = {
    # Base/Clear Scenarios
    'S-01': {'expected': '1 sign, 1 stoplight, 1 regulatory sign classified', 'description': 'A stoplight and a regulatory sign on the street in foggy weather during winter in dim lighting.'},
    'S-02': {'expected': '2 signs, 8 stoplights, 1 regulatory sign classified, 1 street sign classified', 'description': 'An intersection with 8 stoplights, 1 regulatory sign, and 1 street sign in clear weather during spring in normal lighting.'},
    'S-03': {'expected': '2 signs, 8 stoplights, 1 regulatory sign classified, 1 street sign classified', 'description': 'An intersection with 8 stoplights, 1 regulatory sign, and 1 street sign in clear weather during winter in normal lighting.'},
    'S-04': {'expected': '0 signs, 0 stoplights', 'description': 'A street with no stoplights or signs in foggy weather during winter in normal lighting.'},
    'S-05': {'expected': '0 signs, 0 stoplights', 'description': 'A street with no stoplights or signs in clear weather during summer in normal lighting.'},
    'S-06': {'expected': '0 signs, 0 stoplights', 'description': 'A street with no stoplights or signs in clear weather during fall in normal lighting.'},
    'S-07': {'expected': '1 sign, 0 stoplights, 1 stop sign classified', 'description': 'A street with 1 stop sign in clear weather during spring in dim lighting.'},
    'S-08': {'expected': '1 sign, 0 stoplights, 1 regulatory sign classified', 'description': 'A street with 1 regulatory sign in clear weather during summer in normal lighting.'},
    'S-09': {'expected': '1 sign, 0 stoplights, 1 warning sign classified', 'description': 'A street with 1 warning sign in clear weather during summer in normal lighting.'},
    'S-10': {'expected': '1 sign, 0 stoplights, 1 street sign classified', 'description': 'A street with 1 street sign in clear weather during winter in normal lighting.'},
    'S-11': {'expected': '1 sign, 0 stoplights, 1 speed limit sign classified', 'description': 'A street with 1 speed limit sign in clear weather during fall in normal lighting.'},
    'S-12': {'expected': '0 signs, 3 stoplights', 'description': 'An image of 3 stoplights in clear weather during summer in normal lighting.'},
    'S-13': {'expected': '4 signs, 0 stoplights, 1 stop sign classified, 2 street signs classified, 1 speed limit sign classified', 'description': 'A street with 1 stop sign, 2 street signs, and 1 speed limit sign in clear weather during summer in normal lighting.'},
    'S-14': {'expected': '1 sign, 0 stoplights, 1 stop sign classified', 'description': 'A street with 1 stop sign in rainy weather during fall in normal lighting.'},
    # Augmented Scenarios
    'S-15': {'expected': '2 signs, 8 stoplights, 1 regulatory sign classified, 1 street sign classified', 'description': 'Augmented S-02: Sepia at normal brightness.'},
    'S-16': {'expected': '2 signs, 8 stoplights, 1 regulatory sign classified, 1 street sign classified', 'description': 'Augmented S-03: Sharpen at low brightness.'},
    'S-17': {'expected': '0 signs, 0 stoplights', 'description': 'Augmented S-04: Low noise with bright lighting.'},
    'S-18': {'expected': '0 signs, 0 stoplights', 'description': 'Augmented S-05: Emboss at normal brightness.'},
    'S-19': {'expected': '0 signs, 0 stoplights', 'description': 'Augmented S-06: Low blurriness at normal brightness.'},
    'S-20': {'expected': '1 sign, 0 stoplights, 1 stop sign classified', 'description': 'Augmented S-07: Low brightness.'},
    'S-21': {'expected': '1 sign, 0 stoplights, 1 regulatory sign classified', 'description': 'Augmented S-08: Grayscale at normal brightness.'},
    'S-22': {'expected': '1 sign, 0 stoplights, 1 warning sign classified', 'description': 'Augmented S-09: High noise.'},
    'S-23': {'expected': '0 signs, 0 stoplights', 'description': 'Augmented S-10: High pixelation'},
    'S-24': {'expected': '1 sign, 0 stoplights, 1 regulatory sign correctly classified', 'description': 'Augmented S-11: Emboss and high brightness.'},
    'S-25': {'expected': '0 signs, 3 stoplights', 'description': 'Augmented S-12: Invert at normal brightness.'},
    'S-26': {'expected': '4 signs, 0 stoplights, 1 stop sign classified, 2 street signs classified, 1 speed limit sign classified', 'description': 'Augmented S-13: Grayscale with low pixelation at normal brightness.'},
    'S-27': {'expected': '1 sign, 0 stoplights, 1 stop sign classified', 'description': 'Augmented S-14: Blur at normal brightness.'}
}


# --- HELPER FUNCTIONS (adapted from original to handle signs/stoplights) ---

def parse_expected_result(expected_str):
    """
    Parses the complex expected result string into a structured dictionary.
    Example: '1 sign, 1 stoplight, 1 regulatory sign classified'
    """
    expected = {
        'total_signs': 0,
        'total_stoplights': 0,
        'stop_sign': 0,
        'regulatory_sign': 0,
        'street_sign': 0,
        'warning_sign': 0,
        'speed_limit_sign': 0,
    }

    # Extract total signs and stoplights
    sign_match = re.search(r'(\d+)\s+signs?', expected_str)
    stoplight_match = re.search(r'(\d+)\s+stoplights?', expected_str)

    if sign_match:
        expected['total_signs'] = int(sign_match.group(1))
    if stoplight_match:
        expected['total_stoplights'] = int(stoplight_match.group(1))

    # Extract classified signs
    expected['stop_sign'] = len(re.findall(
        r'stop sign classified', expected_str))
    expected['regulatory_sign'] = len(re.findall(
        r'regulatory sign classified', expected_str))
    expected['street_sign'] = len(re.findall(
        r'street sign classified', expected_str))
    expected['warning_sign'] = len(re.findall(
        r'warning sign classified', expected_str))
    expected['speed_limit_sign'] = len(re.findall(
        r'speed limit sign classified', expected_str))

    # Ensure total signs is the sum of classified signs if individual classifications exist
    # This is a heuristic to handle cases where 'X signs detected' is missing or ambiguous
    if expected['total_signs'] == 0 and sum([expected[k] for k in expected if k.endswith('_sign')]) > 0:
        expected['total_signs'] = sum([expected[k]
                                      for k in expected if k.endswith('_sign')])

    return expected


def check_signs_and_lights(all_objects):
    """
    Filters objects for signs and stoplights and performs classification.
    Returns a dictionary of counts.
    """
    detected = {
        'total_signs': 0,
        'total_stoplights': 0,
        'stop_sign': 0,
        'regulatory_sign': 0,
        'street_sign': 0,
        'warning_sign': 0,
        'speed_limit_sign': 0,
    }

    traffic_related_objects = []

    # Map Google Vision labels to test case classifications
    sign_map = {
        'stop sign': 'stop_sign',
        # General classification for non-specific signs
        'traffic sign': 'regulatory_sign',
        'road sign': 'street_sign',      # Heuristic for street names
        'warning sign': 'warning_sign',
        'speed limit sign': 'speed_limit_sign',
        'street sign': 'street_sign'
    }

    # Iterate through all detected objects
    for obj in all_objects:
        name = obj.name.lower()

        # 1. Stoplights
        if 'traffic light' in name or 'stoplight' in name:
            detected['total_stoplights'] += 1
            traffic_related_objects.append(obj)

        # 2. Signs (using heuristics/labeling)
        elif 'sign' in name or 'stop' in name:
            # Attempt to classify the sign
            classified_as = None
            for key, classification in sign_map.items():
                if key in name:
                    classified_as = classification
                    break

            # If a specific classification was found, count it
            if classified_as:
                detected[classified_as] += 1
                detected['total_signs'] += 1
                traffic_related_objects.append(obj)
            elif 'sign' in name:
                # Catch general signs if classification failed
                detected['total_signs'] += 1
                traffic_related_objects.append(obj)

    # Note: Google Vision API labels are not strictly controlled like custom models,
    # so this classification is based on common output labels.

    return detected, traffic_related_objects


def check_test_case_s_pass(expected, detected):
    """
    Compares expected and detected sign/stoplight counts and classifications.
    Returns True if all expected counts match detected counts.
    """
    # Check total signs and stoplights
    if detected['total_signs'] != expected['total_signs']:
        return False

    if detected['total_stoplights'] != expected['total_stoplights']:
        return False

    # Check specific sign classifications
    sign_keys = ['stop_sign', 'regulatory_sign',
                 'street_sign', 'warning_sign', 'speed_limit_sign']
    for key in sign_keys:
        if detected[key] != expected[key]:
            return False

    return True


def show_input_image(image_path, test_id, description, expected_str):
    """Display the input image before processing"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(
        f"INPUT IMAGE: {test_id}\n{description}\nExpected: {expected_str}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()


def show_result_image(output_image_path, test_id, expected_str, detected_dict, passed):
    """Display the result image with bounding boxes"""
    img = mpimg.imread(output_image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"

    # Format detected counts for display
    detected_signs = f"{detected_dict['total_signs']} Signs"
    detected_stoplights = f"{detected_dict['total_stoplights']} Lights"

    ax.set_title(f"RESULT: {test_id} - {status}\nExpected: {expected_str}\nDetected: {detected_signs}, {detected_stoplights}",
                 fontsize=12, fontweight='bold', color=color)
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)  # Show for 3 seconds
    plt.close()


def show_before_after(input_image_path, output_image_path, test_id, description, expected_str, detected_dict, passed):
    """Display input and output images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Input image (left)
    img_input = mpimg.imread(input_image_path)
    axes[0].imshow(img_input)
    axes[0].set_title(
        f"INPUT: {test_id}\n{description}\nExpected: {expected_str}", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Output image with bounding boxes (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)
    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"

    # Format detected counts for display
    detected_signs = f"{detected_dict['total_signs']} Signs"
    detected_stoplights = f"{detected_dict['total_stoplights']} Lights"

    axes[1].set_title(f"OUTPUT: Detected: {detected_signs}, {detected_stoplights}\n{status}",
                      fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(4)  # Show for 4 seconds
    plt.close()


def run_test_s(test_id, client, images_dir, output_dir):
    """Execute a single signs/stoplights test and return results"""
    test_info = TEST_CASES_S.get(
        test_id, {'expected': '0 signs, 0 stoplights', 'description': 'Unknown'})
    expected_str = test_info['expected']
    description = test_info['description']
    expected_dict = parse_expected_result(expected_str)

    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Find image
    image_path = find_image_path(images_dir, test_id)
    if not image_path:
        return None

    # Show input image FIRST before processing
    print(f"\n   Showing input image...")
    show_input_image(image_path, test_id, description, expected_str)

    # Call Vision API
    print(f"   Calling Google Vision API...")
    with open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    all_objects = response.localized_object_annotations

    # Filter for signs and stoplights and get detected counts
    detected_dict, traffic_related_objects = check_signs_and_lights(
        all_objects)

    # Determine pass/fail
    test_passed = check_test_case_s_pass(expected_dict, detected_dict)

    # Generate output image with bounding boxes
    output_image_path = os.path.join(output_dir, f'{test_id}_result.jpg')
    # Use only the traffic-related objects for drawing boxes
    draw_boxes_signs(image_path, traffic_related_objects, output_image_path)

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

    # Calculate total expected objects (integer sum) to satisfy test_utils.py
    # This ensures expected_count is an integer for subtraction in create_json_output.
    total_expected_objects = expected_dict['total_signs'] + \
        expected_dict['total_stoplights']

    # Call create_json_output using the integer count and positional arguments
    # (Must pass traffic_related_objects as the second positional argument)
    json_data = create_json_output(
        test_id,
        traffic_related_objects,
        output_files,
        timing_info=timing_info,
        expected_count=total_expected_objects,  # Passed as integer
        test_passed=test_passed
    )

    # Manually re-insert the descriptive string into the JSON output (Workaround)
    if 'detection_results' in json_data:
        json_data['detection_results']['expected_description'] = expected_str

    save_json_output(json_data, output_files['output_json'])

    return {
        'test_id': test_id,
        'description': description,
        'expected_str': expected_str,
        'expected_dict': expected_dict,
        'detected_dict': detected_dict,
        'passed': test_passed,
        'input_image': image_path,
        'output_image': output_image_path,
        'object_details': [(p.name, p.score) for p in traffic_related_objects],
        'all_objects': [(o.name, o.score) for o in all_objects],
        'duration': duration
    }


def main():
    base_path = script_dir
    # Assuming signs images are in a subdirectory like 'images/ai_tests/signs' or just 'images/ai_tests'
    images_dir = os.path.join(base_path, 'images', 'ai_tests')
    output_dir = setup_output_directory(base_path)

    print()
    print("=" * 80)
    print("CMPE 187 - AI SIGNS & STOPLIGHTS DETECTION TEST SUITE DEMO")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(TEST_CASES_S)}")
    print(f"Output Directory: {output_dir}")
    print()

    # Initialize API client
    print("Initializing Google Cloud Vision API...")
    client = vision.ImageAnnotatorClient()
    print("API ready.\n")

    results = []
    test_ids = sorted(TEST_CASES_S.keys())

    # Start overall timer
    overall_start_time = time.time()

    for i, test_id in enumerate(test_ids, 1):
        test_info = TEST_CASES_S[test_id]

        print("=" * 80)
        print(f"TEST {i}/{len(test_ids)}: {test_id}")
        print("=" * 80)

        # INPUT
        print(f"\n{Colors.CYAN}[INPUT]{Colors.END}")
        print(f"   Test ID:     {test_id}")
        print(f"   Description: {test_info['description']}")

        # EXPECTED OUTPUT
        print(f"\n{Colors.CYAN}[EXPECTED OUTPUT]{Colors.END}")
        print(f"   Expected: {test_info['expected']}")

        # Run test (this shows input image first, then processes)
        result = run_test_s(test_id, client, images_dir, output_dir)

        if result is None:
            print(f"\n{Colors.RED}ERROR: Image not found for {test_id}{Colors.END}")
            continue

        detected_dict = result['detected_dict']
        detected_signs = detected_dict['total_signs']
        detected_lights = detected_dict['total_stoplights']

        # ACTUAL OUTPUT
        print(f"\n{Colors.CYAN}[ACTUAL OUTPUT]{Colors.END}")
        print(f"   Total Signs Detected:    {detected_signs}")
        print(f"   Total Lights Detected:   {detected_lights}")
        print(f"   Total Objects from API: {len(result['all_objects'])}")

        # Print Classification Breakdown
        print(f"\n   Sign Classification:")
        print(f"     Stop:       {detected_dict['stop_sign']}")
        print(f"     Regulatory: {detected_dict['regulatory_sign']}")
        print(f"     Street:     {detected_dict['street_sign']}")
        print(f"     Warning:    {detected_dict['warning_sign']}")
        print(f"     Speed Limit:{detected_dict['speed_limit_sign']}")

        if result['object_details']:
            print(f"\n   Traffic objects detected:")
            for j, (name, score) in enumerate(result['object_details'], 1):
                print(f"     Object {j}: {name} ({score*100:.1f}% confidence)")

        # RESULT
        print(f"\n{Colors.CYAN}[RESULT]{Colors.END}")
        if result['passed']:
            print(
                f"   {Colors.GREEN}{Colors.BOLD}PASS{Colors.END} - All expected counts and classifications match.")
        else:
            print(
                f"   {Colors.RED}{Colors.BOLD}FAIL{Colors.END} - Mismatch in counts or classifications.")

        print(f"   Duration: {result['duration']:.2f}s")

        # Show before/after comparison
        print(f"\n   Showing result comparison...")
        show_before_after(result['input_image'], result['output_image'],
                          test_id, result['description'],
                          result['expected_str'], detected_dict, result['passed'])

        results.append(result)
        print()

    # Calculate overall execution time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    # Summary
    print("=" * 80)
    print("TEST EXECUTION COMPLETE - SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r['passed'])
    failed = len(results) - passed
    pass_rate = (passed / len(results) * 100) if results else 0

    print(f"\n{'Test ID':<10} {'Result':>10} {'Exp. Signs':>10} {'Det. Signs':>10} {'Exp. Lights':>10} {'Det. Lights':>10}")
    print("-" * 65)

    for r in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if r['passed'] else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{r['test_id']:<10} {status:>10} {r['expected_dict']['total_signs']:>10} {r['detected_dict']['total_signs']:>10} {r['expected_dict']['total_stoplights']:>11} {r['detected_dict']['total_stoplights']:>11}")

    print("-" * 65)
    print(f"\nTotal:   {len(results)} tests")
    print(f"Passed:  {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed:  {Colors.RED}{failed}{Colors.END}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    # Display overall execution time
    minutes = int(overall_duration // 60)
    seconds = overall_duration % 60
    if minutes > 0:
        print(f"\nTotal Execution Time: {minutes}m {seconds:.2f}s")
    else:
        print(f"\nTotal Execution Time: {seconds:.2f}s")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
