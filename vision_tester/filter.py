

def filter_objects(objects, text_filter=None):
    
    def apply_filter(obj):
        return text_filter in obj.name

    if text_filter is not None:
        output = list(filter(apply_filter, objects))
    else:
        output = objects

    text_arr = []
    text_arr.append(f"Number of objects found: {len(objects)}")
    for object_ in output:
        text_arr.append(f"{object_.name} ({object_.score})")


    print(f"Number of objects found: {len(objects)}")
    for i, object_ in enumerate(output):
        print(f"\n{i} : {object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
            print(f" - ({vertex.x}, {vertex.y})")
    
    return output, text_arr


