"""
Test Case ID: BVA-012
Category: Boundary Value Analysis
Description: Exact Count of 12 Pedestrians (Large Group Min+1). Twelve people in a busy intersection with crowd conditions, multiple occlusions.

Test Input:
[IMAGE: Twelve people at crosswalk, overhead view]

Input Categories:
- Environmental Conditions: Daylight
- Distance Range: Medium (overhead view)
- Occlusion Level: Moderate (multiple occlusions)
- Group Size: Large Group (12 people)

Expected Result:
- Detection count = 12
- Detection rate: ≥85% (at least 11/12 detected)
- Count accuracy: Within ±±20% (10-14) tolerance for large group
- Large group classification
- Confidence >0.70 (overhead view with moderate occlusions)

Success Criteria:
- PASS: ≥85% detection rate, count within tolerance, confidence >0.70
- FAIL: <85% detection rate or count error exceeds tolerance

Actual Result: [To be filled during testing]
Pass/Fail: [To be determined]
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import vision
from tests.test_utils import (
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


def test_bva_012():
    """
    Test BVA-012: Exact Count of 12 Pedestrians (Large Group Min+1)

    This test verifies detection of 12 people with overhead view and occlusions.
    """
    TEST_ID = 'BVA-012'

    # Record start time
    start_time = time.time()
    start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

    # Set up the image path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_path, 'images')

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
    actual_people = 12  # Ground truth
    detected_people = len(people_detected)

    metrics = calculate_metrics(actual_people, detected_people, 'large')

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

    print(f"\nTEST CASE BVA-012: {status_message}\n")

    # Generate output image with bounding boxes (only for people)
    output_dir = setup_output_directory(base_path)

    output_path = os.path.join(output_dir, f'{TEST_ID}_result.jpg')
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Prepare test configuration for JSON output
    test_config = {
        'test_id': TEST_ID,
        'test_name': 'Exact Count of 12 Pedestrians (Large Group Min+1)',
        'category': 'Boundary Value Analysis',
        'actual_people': actual_people,
        'input_categories': {
            'environmental_conditions': 'Daylight',
            'distance_range': 'Medium (overhead view)',
            'occlusion_level': 'Moderate (multiple occlusions)',
            'group_size': 'Large Group (12 people)'
        },
        'expected_results': {
            'actual_people_in_scene': 12,
            'detection_rate_threshold': 85,
            'count_tolerance': metrics['count_tolerance']
        },
        'comparison': {
            'detection_rate_met': metrics['detection_rate'] >= 85,
            'count_within_tolerance': metrics['count_within_tolerance'],
            'expected_vs_actual_count': {
                'expected': 12,
                'actual': detected_people,
                'difference': metrics['count_error'],
                'exact_match': detected_people == 12
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
            f"Test PASSED: Expected 12 people, detected {detected_people} people. "
            f"Detection rate: {metrics['detection_rate']:.1f}% (≥85% required). "
            f"Count error: {metrics['count_error']} (within ±{metrics['count_tolerance']} tolerance for large groups). "
            f"Meets functional requirements."
            if test_passed else
            f"Test FAILED: Expected 12 people, detected {detected_people} people. "
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
    test_bva_012()
