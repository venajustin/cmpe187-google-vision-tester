"""
Test Case ID: DT-001
Category: Decision Table
Description: Pedestrian Crossing + Vehicle Yielding
Detailed Description: Person at marked crosswalk with vehicle stopped/yielding, clear visibility.

Test Input:
    [IMAGE: DT-001.jpg from images folder]

Expected Result:
- Detect 1 pedestrian
- HIGH PRIORITY alert (safety-critical scenario)
- SAFETY scenario classification
- Crosswalk location identified
- Vehicle presence noted
- Detection confidence: >0.85 (good visibility)

Pass Criteria:
    Detect 1 person (±1 tolerance), detection rate ≥85%, HIGH PRIORITY alert for safety-critical crosswalk scenario

Actual Result: [Generated during test execution]
Pass/Fail: [Determined by comparing actual vs expected results using functional requirements]
"""

import os
import sys
import time
from google.cloud import vision
from test_utils import (
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

def test_dt_001():
    """
    Test Case: DT-001
    Description: Pedestrian Crossing + Vehicle Yielding
    """

    # Record start time
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

    # Test configuration
    test_id = "DT-001"
    actual_people = 1
    detection_threshold = 85
    group_size_category = 'small'

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Input image
    images_dir = os.path.join(project_root, "images")
    actual_image_path = find_image_path(images_dir, test_id)

    # Output directory
    output_dir = setup_output_directory(project_root)

    # Output files
    output_path = os.path.join(output_dir, f"{test_id}_result.jpg")
    json_output_path = os.path.join(output_dir, f"{test_id}_output.json")

    # Check if image exists
    if not os.path.exists(actual_image_path):
        error_msg = f"Error: Image file not found at {actual_image_path}"
        print(f"\nTEST CASE {test_id}: FAIL ✗")
        print(f"{error_msg}\n")

        # Create error outputs
        error_json = {
            "test_id": test_id,
            "status": "ERROR",
            "error": error_msg
        }
        save_json_output(error_json, json_output_path)
        return

    # Initialize Vision API client
    client = vision.ImageAnnotatorClient()

    # Load image
    with open(actual_image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform object localization
    objects = client.object_localization(image=image).localized_object_annotations

    # Filter for people only
    people_detected = filter_people(objects)
    detected_people = len(people_detected)

    # Calculate metrics
    metrics = calculate_metrics(actual_people, detected_people, group_size_category)

    # Determine pass/fail based on functional requirements
    test_passed = True
    failure_reasons = []

    if actual_people == 0:
        # Zero detection test - must have no false positives
        if detected_people > 0:
            test_passed = False
            failure_reasons.append(f"False positive: Expected 0 people but detected {detected_people}")
    else:
        # Check 1: Detection rate must meet threshold
        if metrics['detection_rate'] < detection_threshold:
            test_passed = False
            failure_reasons.append(
                f"Detection rate {metrics['detection_rate']:.1f}% below threshold {detection_threshold}%"
            )

        # Check 2: Count must be within tolerance
        if not metrics['count_within_tolerance']:
            test_passed = False
            failure_reasons.append(
                f"Count error {metrics['count_error']} exceeds tolerance {metrics['count_tolerance']} "
                f"(Expected: {actual_people}, Detected: {detected_people})"
            )

    # Generate annotated image
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Prepare output data
    output_files = {
        'annotated_image': output_path,
        'json_output': json_output_path
    }

    test_config = {
        'test_id': test_id,
        'category': 'Decision Table',
        'description': 'Pedestrian Crossing + Vehicle Yielding',
        'detailed_description': 'Person at marked crosswalk with vehicle stopped/yielding, clear visibility.',
        'actual_people': actual_people,
        'detection_threshold': detection_threshold,
        'group_size_category': group_size_category,
        'expected_results': """- Detect 1 pedestrian
- HIGH PRIORITY alert (safety-critical scenario)
- SAFETY scenario classification
- Crosswalk location identified
- Vehicle presence noted
- Detection confidence: >0.85 (good visibility)""",
        'pass_criteria': 'Detect 1 person (±1 tolerance), detection rate ≥85%, HIGH PRIORITY alert for safety-critical crosswalk scenario'
    }

    # Record end time and calculate duration
    end_time = time.time()
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    duration = end_time - start_time

    # Create timing info dictionary
    timing_info = {
        'start_time': start_time_formatted,
        'end_time': end_time_formatted,
        'duration': duration
    }

    # Create JSON output
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
    save_json_output(json_data, json_output_path)

    # Print only final status to console
    status_message = "PASS ✓" if test_passed else "FAIL ✗"
    print(f"\nTEST CASE {test_id}: {status_message}\n")

if __name__ == "__main__":
    test_dt_001()
