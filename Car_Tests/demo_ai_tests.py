#!/usr/bin/env python3
"""
AI Test Suite Demo Script for CMPE 187
=====================================

Runs all 28 AI test cases and displays for each:
- Input: Shows the input image first
- Expected Output: Expected number of vehicles
- Actual Output: Detected count + annotated image with bounding boxes
- Result: PASS or FAIL

Usage:
    python3 demo_ai_tests.py
"""

import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from google.cloud import vision
from tests.ai_tests.test_utils import (
    draw_bounding_boxes,
    find_image_path,
    filter_vehicles,
    create_json_output,
    save_json_output,
    setup_output_directory
)


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
        'c-01':  {'expected': 6,  'description': '1 to 5 cars, far distance, overlapping, car type: vehicles'},
        'c-02':  {'expected': 6,  'description': 'Augmented C-01: Heavy Pixelation, 1 to 5 cars, far distance, overlapping, car type: vehilces'},
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


def show_input_image(image_path, test_id, description, expected_count):
    """Display the input image before processing"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"INPUT IMAGE: {test_id}\n{description}\nExpected Vehicles: {expected_count}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
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


def show_before_after(input_image_path, output_image_path, test_id, description, expected, detected, passed):
    """Display input and output images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Input image (left)
    img_input = mpimg.imread(input_image_path)
    axes[0].imshow(img_input)
    axes[0].set_title(f"INPUT: {test_id}\n{description}\nExpected: {expected} vehicles", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Output image with bounding boxes (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)
    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"
    axes[1].set_title(f"OUTPUT: Detected: {detected} vehicles\n{status}", fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(4)  # Show for 4 seconds
    plt.close()


def run_test(test_id, client, images_dir, output_dir):
    """Execute a single test and return results"""
    test_info = TEST_CASES.get(test_id, {'expected': 0, 'description': 'Unknown'})
    expected_count = test_info['expected']
    description = test_info['description']

    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    # Find image
    image_path = find_image_path(images_dir, test_id)
    if not image_path:
        return None

    # Show input image FIRST before processing
    print(f"\n  Showing input image...")
    show_input_image(image_path, test_id, description, expected_count)

    # Call Vision API
    print(f"  Calling Google Vision API...")
    with open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    all_objects = response.localized_object_annotations

    # Filter for vehicles
    vehicles_detected = filter_vehicles(all_objects)
    detected_count = len(vehicles_detected)

    # Determine pass/fail
    test_passed = detected_count == expected_count

    # Generate output image with bounding boxes
    output_image_path = os.path.join(output_dir, f'{test_id}_result.jpg')
    draw_bounding_boxes(image_path, vehicles_detected, output_image_path)

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
    json_data = create_json_output(test_id, vehicles_detected, output_files, timing_info, expected_count, test_passed)
    save_json_output(json_data, output_files['output_json'])

    return {
        'test_id': test_id,
        'description': description,
        'expected': expected_count,
        'detected': detected_count,
        'passed': test_passed,
        'input_image': image_path,
        'output_image': output_image_path,
        'vehicle_details': [(p.name, p.score) for p in vehicles_detected],
        'all_objects': [(o.name, o.score) for o in all_objects],
        'duration': duration
    }


def main():
    base_path = script_dir
    images_dir = os.path.join(base_path, 'images', 'ai_tests')
    output_dir = setup_output_directory(base_path)

    print()
    print("=" * 80)
    print("CMPE 187 - AI VEHICLE DETECTION TEST SUITE DEMO")
    print("=" * 80)
    print(f"\nTotal Test Cases: {len(TEST_CASES)}")
    print(f"Output Directory: {output_dir}")
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
        print(f"  Expected Vehicle Count: {test_info['expected']}")

        # Run test (this shows input image first, then processes)
        result = run_test(test_id, client, images_dir, output_dir)

        if result is None:
            print(f"\n{Colors.RED}ERROR: Image not found for {test_id}{Colors.END}")
            continue

        # ACTUAL OUTPUT
        print(f"\n{Colors.CYAN}[ACTUAL OUTPUT]{Colors.END}")
        print(f"  Detected Vehicle Count: {result['detected']}")
        print(f"  All Objects Detected:  {len(result['all_objects'])}")

        if result['all_objects']:
            print(f"\n  Objects from API:")
            for name, score in result['all_objects']:
                is_vehicle = name.lower() in ['car', 'truck']
                marker = " <-- VEHICLE" if is_vehicle else ""
                print(f"    - {name}: {score*100:.1f}%{marker}")

        if result['vehicle_details']:
            print(f"\n  Cars detected:")
            for j, (name, score) in enumerate(result['vehicle_details'], 1):
                print(f"    Person {j}: {score*100:.1f}% confidence")

        # RESULT
        print(f"\n{Colors.CYAN}[RESULT]{Colors.END}")
        if result['passed']:
            print(f"  {Colors.GREEN}{Colors.BOLD}PASS{Colors.END} - Expected {result['expected']}, Detected {result['detected']}")
        else:
            print(f"  {Colors.RED}{Colors.BOLD}FAIL{Colors.END} - Expected {result['expected']}, Detected {result['detected']}")

        print(f"  Duration: {result['duration']:.2f}s")

        # Show before/after comparison
        print(f"\n  Showing result comparison...")
        show_before_after(result['input_image'], result['output_image'],
                          test_id, result['description'],
                          result['expected'], result['detected'], result['passed'])

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

    print(f"\n{'Test ID':<10} {'Expected':>10} {'Detected':>10} {'Result':>10}")
    print("-" * 45)

    for r in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if r['passed'] else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{r['test_id']:<10} {r['expected']:>10} {r['detected']:>10} {status:>20}")

    print("-" * 45)
    print(f"\nTotal:  {len(results)} tests")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed: {Colors.RED}{failed}{Colors.END}")
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
