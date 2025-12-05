
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

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"


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
    objects = localize_objects(img)

    found_objects = []
    detected_objects = []
    # process stats
    for o in objects:
        if o.name in test['objects']:
            detected_objects.push(o)
            found_objects.push({
                    "result": o,
                    "matching_expected": None # TODO
                })
            

    output_image_path = os.path.join(OUTPUT_DIR, f"{test['name']}_result.jpg")
    
    draw_bounding_boxes(img_path, detected_objects, output_image_path)

    # display output


    # print results


def run_suite():
    with open(INPUT_DIR + "/input_expected.json") as f:
        tests = json.load(f)
        for test in tests["input"]:
            run_test(test)

run_suite()




