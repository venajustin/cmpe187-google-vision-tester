#!/usr/bin/env python3
"""
Script to run a single test case and display the raw Google Cloud Vision API response.

Usage:
    python3 show_api_response.py [TEST_ID]

Example:
    python3 show_api_response.py BVA-008
"""

import sys
import os
from google.cloud import vision
import json

def find_image_path(test_id, base_path='People_Tests'):
    """Find image file with various extensions."""
    extensions = ['.jpg', '.jpeg', '.png']
    for ext in extensions:
        image_path = os.path.join(base_path, 'images', f'{test_id}{ext}')
        if os.path.exists(image_path):
            return image_path
    return None

def show_raw_api_response(test_id):
    """Run Vision API and display raw response for a test case."""

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir.endswith('People_Tests'):
        base_path = os.path.join(script_dir, 'People_Tests')
    else:
        base_path = script_dir

    # Find image
    image_path = find_image_path(test_id, base_path)
    if not image_path:
        print(f"❌ Error: Could not find image for {test_id}")
        print(f"   Looked in: {os.path.join(base_path, 'images')}")
        return

    print("=" * 80)
    print(f"Google Cloud Vision API - Raw Response for {test_id}")
    print("=" * 80)
    print(f"\nImage: {image_path}")
    print(f"Exists: {os.path.exists(image_path)}")
    print("\n" + "-" * 80)
    print("Calling Vision API...")
    print("-" * 80 + "\n")

    # Initialize Vision API client
    client = vision.ImageAnnotatorClient()

    # Load image
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Call Object Localization
    response = client.object_localization(image=image)

    # Display raw response
    print("RAW API RESPONSE:")
    print("=" * 80)

    if response.error.message:
        print(f"API Error: {response.error.message}")
        return

    # Show total objects detected
    total_objects = len(response.localized_object_annotations)
    print(f"\nTotal Objects Detected: {total_objects}\n")

    # Display each object in detail
    for idx, obj in enumerate(response.localized_object_annotations, 1):
        print(f"\n{'─' * 80}")
        print(f"Object #{idx}")
        print(f"{'─' * 80}")
        print(f"  Name:       {obj.name}")
        print(f"  Score:      {obj.score:.4f} ({obj.score * 100:.2f}%)")
        print(f"  MID:        {obj.mid}")

        # Bounding box vertices
        print(f"\n  Bounding Box (normalized coordinates):")
        vertices = obj.bounding_poly.normalized_vertices
        for v_idx, vertex in enumerate(vertices):
            print(f"    Vertex {v_idx}: (x={vertex.x:.4f}, y={vertex.y:.4f})")

    # Filter to people only
    people_objects = [
        obj for obj in response.localized_object_annotations
        if obj.name.lower() in ['person', 'people', 'pedestrian']
    ]

    print("\n" + "=" * 80)
    print("PEOPLE DETECTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal objects detected: {total_objects}")
    print(f"People detected:        {len(people_objects)}")

    if people_objects:
        print(f"\nPeople objects:")
        for idx, obj in enumerate(people_objects, 1):
            print(f"  {idx}. {obj.name} (confidence: {obj.score * 100:.2f}%)")

    # Show non-people objects if any
    non_people = [
        obj for obj in response.localized_object_annotations
        if obj.name.lower() not in ['person', 'people', 'pedestrian']
    ]

    if non_people:
        print(f"\nNon-people objects detected:")
        for idx, obj in enumerate(non_people, 1):
            print(f"  {idx}. {obj.name} (confidence: {obj.score * 100:.2f}%)")

    print("\n" + "=" * 80)

    # Export to JSON for detailed analysis
    json_output = {
        'test_id': test_id,
        'image_path': image_path,
        'total_objects_detected': total_objects,
        'people_count': len(people_objects),
        'objects': []
    }

    for obj in response.localized_object_annotations:
        obj_dict = {
            'name': obj.name,
            'score': obj.score,
            'mid': obj.mid,
            'bounding_poly': {
                'normalized_vertices': [
                    {'x': v.x, 'y': v.y} for v in obj.bounding_poly.normalized_vertices
                ]
            }
        }
        json_output['objects'].append(obj_dict)

    # Save JSON
    json_path = os.path.join(base_path, 'results', f'{test_id}_raw_api_response.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"\nRaw API response saved to: {json_path}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    # Get test ID from command line or use default
    test_id = sys.argv[1] if len(sys.argv) > 1 else 'BVA-008'
    show_raw_api_response(test_id)
