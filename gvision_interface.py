from io import BytesIO
def localize_objects(img): 

    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    buffer = BytesIO()
    img.save(buffer, img.format)
    image = vision.Image(content=buffer.getvalue())

    objects = client.object_localization(image=image).localized_object_annotations

    return objects
    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        print(f"\n{object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
            print(f" - ({vertex.x}, {vertex.y})")
