"""
Test Case ID: BVA-001
Category: Boundary Value Analysis
Description: Zero Pedestrians (Empty Road). Empty urban street with no
people present, clear visibility, good daylight lighting.

Test Input:
[IMAGE: Empty urban street with buildings, trees, clear road, no pedestrians]

Input Categories:
- Environmental Conditions: Daylight
- Distance Range: N/A (no people)
- Occlusion Level: N/A (no people)
- Group Size: Individual (0 people)

Expected Result:
- Detection count = 0
- No false positives (0 non-persons detected as persons)
- No bounding boxes generated
- Detection rate: N/A (0 actual persons)

Success Criteria:
- PASS: 0 people detected, 0% false positive rate
- FAIL: Any people detected (false positives)

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
    create_json_output,
    save_json_output,
    print_detection_summary,
    setup_output_directory,
    format_duration,
    TeeOutput
)


def test_bva_001():
    """
    Test BVA-001: Zero Pedestrians (Empty Road)

    This test verifies that the Google Vision API correctly identifies
    an empty street with no pedestrians present.
    """
    TEST_ID = 'BVA-001'

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
    actual_people = 0  # Ground truth: 0 people in scene
    detected_people = len(people_detected)
    false_positives = detected_people  # Any detection is a false positive

    # Determine pass/fail based on criteria
    if detected_people == 0:
        test_passed = True
        status_message = "PASS ✓"
    else:
        test_passed = False
        status_message = "FAIL ✗"

    # Print only the final status
    print(f"\nTEST CASE BVA-001: {status_message}\n")

    # Generate output image with bounding boxes (only for people)
    output_dir = setup_output_directory(base_path)

    output_path = os.path.join(output_dir, f'{TEST_ID}_result.jpg')
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Prepare test configuration for JSON output
    test_config = {
        'test_id': TEST_ID,
        'test_name': 'Zero Pedestrians (Empty Road)',
        'category': 'Boundary Value Analysis',
        'actual_people': actual_people,
        'input_categories': {
            'environmental_conditions': 'Daylight',
            'distance_range': 'N/A (no people)',
            'occlusion_level': 'N/A (no people)',
            'group_size': 'Individual (0 people)'
        },
        'expected_results': {
            'actual_people_in_scene': 0,
            'detection_count': 0,
            'false_positives': 0,
            'bounding_boxes': 0
        },
        'comparison': {
            'detection_count_match': detected_people == 0,
            'no_false_positives': false_positives == 0,
            'expected_vs_actual': {
                'expected_people': 0,
                'detected_people': detected_people,
                'exact_match': detected_people == 0
            }
        },
        'test_reason': (
            "Test PASSED: Expected 0 people, detected 0 people. No false positives detected."
            if test_passed else
            f"Test FAILED: Expected 0 people, but detected {detected_people} people. "
            f"Functional requirement violated: {false_positives} false positive(s) detected. "
            f"Zero pedestrians should result in zero detections."
        )
    }

    # Metrics dictionary for JSON
    metrics = {
        'detection_rate': 0,  # N/A for 0 people
        'count_tolerance': 0,
        'count_error': detected_people,
        'count_within_tolerance': detected_people == 0,
        'false_positives': false_positives,
        'false_negatives': 0
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
        [],
        output_files,
        timing_info
    )

    # Save JSON output
    json_output_path = output_files['output_json']
    save_json_output(json_data, json_output_path)

    return test_passed


if __name__ == "__main__":
    test_bva_001()
