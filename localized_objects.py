import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

import sys

def localize_objects(path, ax, img_size):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        print(f"\n{object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
            print(f" - ({vertex.x}, {vertex.y})")
        draw_polygon(ax,coords,img_size);

def setup_plot():
    fig, ax = plt.subplots()
    return fig, ax

def show_image(ax, image_path):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    return img.shape[1], img.shape[0]  # width, height

def draw_polygon(ax, coords, img_size):
    w, h = img_size
    scaled = [(x * w, y * h) for x, y in coords]
    poly = patches.Polygon(scaled, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(poly)


path = sys.argv[1]
output = sys.argv[2]
fig, ax = setup_plot()
w, h = show_image(ax, path)
localize_objects(path, ax, (w,h))
plt.savefig(output)
