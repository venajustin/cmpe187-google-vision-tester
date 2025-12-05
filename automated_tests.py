
"""
AI Test Suite Demo Script for CMPE 187
=====================================
"""
import json
import os
from vision_tester.gvision_interface import *
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from vision_tester.test_utils import * 
import time
from datetime import datetime

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

def show_before_after(input_image_path, output_image_path, test_id, detected, passed):
    """Display input and output images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Input image (left)
    img_input = mpimg.imread(input_image_path)
    axes[0].imshow(img_input)
    axes[0].set_title(f"INPUT: {test_id}\n", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Output image with bounding boxes (right)
    img_output = mpimg.imread(output_image_path)
    axes[1].imshow(img_output)
    status = "PASS" if passed else "FAIL"
    color = "green" if passed else "red"
    axes[1].set_title(f"OUTPUT: Detected: {detected} objects\n{status}", fontsize=12, fontweight='bold', color=color)
    axes[1].axis('off')

    plt.suptitle(f"Test Case {test_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(4)  # Show for 4 seconds
    plt.close()


def show_input_image(image_path, test_id):
    """Display the input image before processing"""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"INPUT IMAGE: {test_id}\n", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()

def run_test(test):
    print("-------------------------------------------")
    print("| Running test: " + test['name'])
    print("-------------------------------------------")
    # load image
    img_path = os.path.join(INPUT_DIR, test['file'])
    print("Loading image: " + img_path)
    img = Image.open(img_path)

    # display image
    show_input_image(img_path, test['name'])

    # call vision model
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    objects = localize_objects(img)

    end_time = time.time()
    duration = end_time - start_time

    timing_info = {
        'start_time': start_timestamp,
        'end_time': datetime.now().isoformat(),
        'duration': duration
    }
    found_objects = []
    detected_objects = []

    # process stats
    for o in objects:
        if o.name.lower() in test['objects']:
            print( "Object found: ", o.name)
            detected_objects.append(o)
            found_objects.append({
                    "result": o,
                    "matching_expected": None # TODO
                })
            

    output_image_path = os.path.join(OUTPUT_DIR, f"{test['name']}_result.jpg")
    
    draw_bounding_boxes(img_path, detected_objects, output_image_path)

    # display output
    show_before_after(img_path, output_image_path,
                      test['name'],
                      len(detected_objects), True) #todo add back in features

    # print results



def run_suite():
    with open(INPUT_DIR + "/input_expected.json") as f:
        tests = json.load(f)
        for test in tests["input"]:
            run_test(test)

run_suite()




