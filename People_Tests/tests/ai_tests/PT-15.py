"""
Test Case ID: PT-15
Category: AI Generated Test
Description: AI-generated test case for people detection
Expected People: 1
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.cloud import vision
from tests.ai_tests.test_utils import (
    draw_bounding_boxes,
    find_image_path,
    filter_people,
    create_json_output,
    save_json_output,
    setup_output_directory
)


def test_pt_15():
    """
    Test PT-15: AI-generated people detection test
    Expected: 1 person(s)
    """
    TEST_ID = 'PT-15'
    EXPECTED_COUNT = 1

    # Record start time
    start_time = time.time()
    start_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

    # Set up paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images_dir = os.path.join(base_path, 'images', 'ai_tests')

    actual_image_path = find_image_path(images_dir, TEST_ID)

    if not actual_image_path:
        print(f"ERROR: Image file not found for {TEST_ID}")
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
    detected_count = len(people_detected)

    # Exact match - pass only if detected equals expected
    test_passed = detected_count == EXPECTED_COUNT

    status_message = "PASS" if test_passed else "FAIL"
    print(f"\nTEST CASE {TEST_ID}: {status_message} (Expected: {EXPECTED_COUNT}, Detected: {detected_count})\n")

    # Generate output image with bounding boxes
    output_dir = setup_output_directory(base_path)
    output_path = os.path.join(output_dir, f'{TEST_ID}_result.jpg')
    draw_bounding_boxes(actual_image_path, people_detected, output_path)

    # Record end time
    end_time = time.time()
    end_time_formatted = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    duration = end_time - start_time

    timing_info = {
        'start_time': start_time_formatted,
        'end_time': end_time_formatted,
        'duration': duration
    }

    # Create and save JSON output
    output_files = {
        'result_image': output_path,
        'output_json': os.path.join(output_dir, f'{TEST_ID}_output.json')
    }

    json_data = create_json_output(TEST_ID, people_detected, output_files, timing_info, EXPECTED_COUNT, test_passed)
    save_json_output(json_data, output_files['output_json'])

    return test_passed


if __name__ == "__main__":
    test_pt_15()
