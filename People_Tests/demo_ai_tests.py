#!/usr/bin/env python3
"""
AI Test Suite Demo Script for CMPE 187
=====================================

Runs all 28 AI test cases and displays for each:
- Input: Shows the input image first
- Expected Output: Expected number of people
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
    filter_people,
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


def show_input_image(image_path, test_id, description, expected_count):
    """Display the input image before processing"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"INPUT IMAGE: {test_id}\n{description}\nExpected People: {expected_count}", fontsize=14, fontweight='bold')
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
    axes[0].set_title(f"INPUT: {test_id}\n{description}\nExpected: {expected} people", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Output image with bounding boxes (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)
    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"
    axes[1].set_title(f"OUTPUT: Detected: {detected} people\n{status}", fontsize=12, fontweight='bold', color=color)
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

    # Filter for people
    people_detected = filter_people(all_objects)
    detected_count = len(people_detected)

    # Determine pass/fail
    test_passed = detected_count == expected_count

    # Generate output image with bounding boxes
    output_image_path = os.path.join(output_dir, f'{test_id}_result.jpg')
    draw_bounding_boxes(image_path, people_detected, output_image_path)

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
    json_data = create_json_output(test_id, people_detected, output_files, timing_info, expected_count, test_passed)
    save_json_output(json_data, output_files['output_json'])

    return {
        'test_id': test_id,
        'description': description,
        'expected': expected_count,
        'detected': detected_count,
        'passed': test_passed,
        'input_image': image_path,
        'output_image': output_image_path,
        'people_details': [(p.name, p.score) for p in people_detected],
        'all_objects': [(o.name, o.score) for o in all_objects],
        'duration': duration
    }


def main():
    base_path = script_dir
    images_dir = os.path.join(base_path, 'images', 'ai_tests')
    output_dir = setup_output_directory(base_path)

    print()
    print("=" * 80)
    print("CMPE 187 - AI PEOPLE DETECTION TEST SUITE DEMO")
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
        print(f"  Expected People Count: {test_info['expected']}")

        # Run test (this shows input image first, then processes)
        result = run_test(test_id, client, images_dir, output_dir)

        if result is None:
            print(f"\n{Colors.RED}ERROR: Image not found for {test_id}{Colors.END}")
            continue

        # ACTUAL OUTPUT
        print(f"\n{Colors.CYAN}[ACTUAL OUTPUT]{Colors.END}")
        print(f"  Detected People Count: {result['detected']}")
        print(f"  All Objects Detected:  {len(result['all_objects'])}")

        if result['all_objects']:
            print(f"\n  Objects from API:")
            for name, score in result['all_objects']:
                is_person = name.lower() in ['person', 'people', 'pedestrian']
                marker = " <-- PERSON" if is_person else ""
                print(f"    - {name}: {score*100:.1f}%{marker}")

        if result['people_details']:
            print(f"\n  People detected:")
            for j, (name, score) in enumerate(result['people_details'], 1):
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
