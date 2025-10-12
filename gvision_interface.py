from io import BytesIO
def localize_objects(img): 

    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    buffer = BytesIO()
    img.save(buffer, img.format)
    image = vision.Image(content=buffer.getvalue())

    objects = client.object_localization(image=image).localized_object_annotations

    text_arr = []
    text_arr.append(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        text_arr.append(f"{object_.name} ({object_.score})")


    print(f"Number of objects found: {len(objects)}")
    for i, object_ in enumerate(objects):
        print(f"\n{i} : {object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
            print(f" - ({vertex.x}, {vertex.y})")
    
    return objects, text_arr
