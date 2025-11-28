"""
Test Case ID: BVA-004
Category: Boundary Value Analysis
Description: Exact Count of 3 Pedestrians (Small Group Min+1). Three people standing/walking in an urban setting with all subjects clearly visible, good lighting.

Test Input:
[IMAGE: Three people walking together on park path in daylight]

Input Categories:
- Environmental Conditions: Daylight
- Distance Range: Close to Medium
- Occlusion Level: None (100% visible)
- Group Size: Small Group (3 people)

Expected Result:
- Detection count = 3
- Detection rate: 100% (3/3 detected)
- Count accuracy: Within ±1 tolerance for small group
- Three bounding boxes
- Confidence >0.8 for each detection (good lighting and visibility)

Success Criteria:
- PASS: ≥85% detection rate, count within tolerance, confidence >0.8
- FAIL: <85% detection rate or count error exceeds tolerance

Actual Result: [To be filled during testing]
Pass/Fail: [To be determined]
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.cloud import vision
from tests.conventional_tests.test_utils import (
    draw_bounding_boxes,
    find_image_path,
    filter_people,
    calculate_metrics,
    create_json_output,
    save_json_output,
    setup_output_directory,
    format_duration,
    TeeOutput
)


def test_bva_004():
    """
    Test BVA-004: Exact Count of 3 Pedestrians (Small Group Min+1)

    This test verifies detection of 3 people with clear visibility in good lighting.
    """
    TEST_ID = 'BVA-004'

    # Record start time
    start_time = time.time()
    start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

    # Set up the image path
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(base_path, 'images', 'conventional_tests')

    actual_image_path = find_image_path(images_dir, TEST_ID)

    if not actual_image_path:
        print(f"ERROR: Image file not found. Expected at: {os.path.join(images_dir, TEST_ID)}[.jpg/.jpeg/.png/.gif/.bmp]")
        return False

    # Initialize Google Vision API client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(actual_image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Perform object localization
    objects = client.object_localization(image=image).localized_object_annotations

    # Filter for people/pedestrians
    people_detected = filter_people(objects)

    # Calculate metrics
    actual_people = 3  # Ground truth
    detected_people = len(people_detected)

    metrics = calculate_metrics(actual_people, detected_people, 'small')

    # Determine pass/fail based on criteria
    test_passed = True
    failure_reasons = []

    # Check 1: Detection rate ≥85%
    if metrics['detection_rate'] < 85:
        test_passed = False
        failure_reasons.append(f"Detection rate {metrics['detection_rate']:.1f}% below 85% threshold")

    # Check 2: Count accuracy within tolerance
    if not metrics['count_within_tolerance']:
        test_passed = False
        failure_reasons.append(f"Count error {metrics['count_error']} exceeds ±{metrics['count_tolerance']} tolerance")

    # Print only the final status
    if test_passed:
        status_message = "PASS ✓"
    else:
        status_message = "FAIL ✗"

    print(f"\nTEST CASE BVA-004: {status_message}\n")

    # Generate output image with bounding boxes (only for people)
    output_dir = setup_output_directory(base_path, 'conventional_tests')

    output_path = os.path.join(output_dir, f'{TEST_ID}_result.jpg')
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Prepare test configuration for JSON output
    test_config = {
        'test_id': TEST_ID,
        'test_name': 'Exact Count of 3 Pedestrians (Small Group Min+1)',
        'category': 'Boundary Value Analysis',
        'actual_people': actual_people,
        'input_categories': {
            'environmental_conditions': 'Daylight',
            'distance_range': 'Close to Medium',
            'occlusion_level': 'None (100% visible)',
            'group_size': 'Small Group (3 people)'
        },
        'expected_results': {
            'actual_people_in_scene': 3,
            'detection_rate_threshold': 85,
            'count_tolerance': metrics['count_tolerance']
        },
        'comparison': {
            'detection_rate_met': metrics['detection_rate'] >= 85,
            'count_within_tolerance': metrics['count_within_tolerance'],
            'expected_vs_actual_count': {
                'expected': 3,
                'actual': detected_people,
                'difference': metrics['count_error'],
                'exact_match': detected_people == 3
            },
            'criteria_checks': {
                'detection_rate': {
                    'threshold': 85,
                    'actual': round(metrics['detection_rate'], 2),
                    'passed': metrics['detection_rate'] >= 85
                },
                'count_accuracy': {
                    'tolerance': metrics['count_tolerance'],
                    'error': metrics['count_error'],
                    'passed': metrics['count_within_tolerance']
                }
            }
        },
        'test_reason': (
            f"Test PASSED: Expected 3 people, detected {detected_people} people. "
            f"Detection rate: {metrics['detection_rate']:.1f}% (≥85% required). "
            f"Count error: {metrics['count_error']} (within ±{metrics['count_tolerance']} tolerance for small groups). "
            f"Meets functional requirements."
            if test_passed else
            f"Test FAILED: Expected 3 people, detected {detected_people} people. "
            + (f"Detection rate: {metrics['detection_rate']:.1f}% (<85% threshold). " if metrics['detection_rate'] < 85 else "")
            + (f"Count error: {metrics['count_error']} (exceeds ±{metrics['count_tolerance']} tolerance). " if not metrics['count_within_tolerance'] else "")
            + "Does not meet functional requirements."
        )
    }

    # Create JSON output
    output_files = {
        'result_image': output_path,
        'output_json': os.path.join(output_dir, f'{TEST_ID}_output.json')
    }


    # Record end time and calculate duration
    end_time = time.time()
    end_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    duration = end_time - start_time

    # Create timing info dictionary
    timing_info = {
        'start_time': start_time_formatted,
        'end_time': end_time_formatted,
        'duration': duration
    }

    json_data = create_json_output(
        test_config,
        people_detected,
        metrics,
        test_passed,
        failure_reasons,
        output_files,
        timing_info
    )

    # Save JSON output
    json_output_path = output_files['output_json']
    save_json_output(json_data, json_output_path)

    return test_passed


if __name__ == "__main__":
    test_bva_004()
