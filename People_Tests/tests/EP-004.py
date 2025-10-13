"""
Test Case ID: EP-004
Category: Equivalence Partition
Description: Extreme Glare/Blinding Light Class. Scene with sun glare or bright reflections affecting visibility, direct sunlight into the camera. 19 clearly visible people with additional faint figures obscured by glare.

Test Input:
[IMAGE: Street scene with 19 people, intense sun glare, golden lighting]

Input Categories:
- Environmental Conditions: Extreme glare/blinding light
- Distance Range: Close to Medium
- Occlusion Level: Visual degradation from glare
- Group Size: Large Group (19 people)

Expected Result:
- Detection count = 19 (clearly visible people)
- Degraded detection performance due to glare
- Lower confidence acceptable (0.55-0.75)
- Some missed detections acceptable (especially for silhouetted/washed-out figures)
- Bounding boxes may be less precise

Success Criteria:
- PASS: ≥60% detection rate (glare challenge), count within ±20% tolerance for large group
- FAIL: <60% detection rate or count error exceeds tolerance

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


def test_ep_004():
    """
    Test EP-004: Extreme Glare/Blinding Light Class (19 people)

    This test verifies detection with extreme glare/blinding light conditions affecting 19 people.
    """
    TEST_ID = 'EP-004'

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
    actual_people = 19  # Ground truth: 19 clearly visible people (more faint in background but uncountable due to glare)
    detected_people = len(people_detected)

    metrics = calculate_metrics(actual_people, detected_people, 'large')

    # Determine pass/fail based on criteria
    test_passed = True
    failure_reasons = []

    # Check detection rate threshold
    detection_threshold = 60
    if metrics['detection_rate'] < detection_threshold:
        test_passed = False
        failure_reasons.append(f"Detection rate {metrics['detection_rate']:.1f}% below {detection_threshold}% threshold")

    # Check count accuracy within tolerance
    if not metrics['count_within_tolerance']:
        test_passed = False
        failure_reasons.append(f"Count error {metrics['count_error']} exceeds ±{metrics['count_tolerance']} tolerance")

    # Print only the final status
    if test_passed:
        status_message = "PASS ✓"
    else:
        status_message = "FAIL ✗"

    print(f"\nTEST CASE EP-004: {status_message}\n")

    # Generate output image with bounding boxes (only for people)
    output_dir = setup_output_directory(base_path)

    output_path = os.path.join(output_dir, f'{TEST_ID}_result.jpg')
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Prepare test configuration for JSON output
    test_config = {
        'test_id': TEST_ID,
        'test_name': 'Extreme Glare/Blinding Light Class (19 people)',
        'category': 'Equivalence Partition',
        'actual_people': actual_people,
        'input_categories': {
            'environmental_conditions': 'Extreme glare/blinding light',
            'distance_range': 'Close to Medium',
            'occlusion_level': 'Visual degradation from glare',
            'group_size': 'Large Group (19 people)'
        },
        'expected_results': {
            'actual_people_in_scene': 19,
            'detection_rate_threshold': 60,
            'count_tolerance': metrics['count_tolerance']
        },
        'comparison': {
            'detection_rate_met': metrics['detection_rate'] >= 60,
            'count_within_tolerance': metrics['count_within_tolerance'],
            'expected_vs_actual_count': {
                'expected': 19,
                'actual': detected_people,
                'difference': metrics['count_error'],
                'exact_match': detected_people == 19
            },
            'criteria_checks': {
                'detection_rate': {
                    'threshold': 60,
                    'actual': round(metrics['detection_rate'], 2),
                    'passed': metrics['detection_rate'] >= 60
                },
                'count_accuracy': {
                    'tolerance': metrics['count_tolerance'],
                    'error': metrics['count_error'],
                    'passed': metrics['count_within_tolerance']
                }
            }
        },
        'test_reason': (
            f"Test PASSED: Expected 19 people, detected {detected_people} people. "
            f"Detection rate: {metrics['detection_rate']:.1f}% (≥60% required). "
            f"Count error: {metrics['count_error']} (within ±{metrics['count_tolerance']} tolerance for large groups). "
            f"Meets functional requirements for Extreme Glare/Blinding Light Class."
            if test_passed else
            f"Test FAILED: Expected 19 people, detected {detected_people} people. "
            + (f"Detection rate: {metrics['detection_rate']:.1f}% (<60% threshold). " if metrics['detection_rate'] < 60 else "")
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
    test_ep_004()
