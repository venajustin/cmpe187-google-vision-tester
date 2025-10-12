import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from io import BytesIO
from PIL import Image

def label_img(objects, img):

    fig, ax = setup_plot()
    w, h = show_image(ax, img)

    for object_ in objects:
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
        draw_polygon(ax,coords,(w,h))

    bio = BytesIO()
    plt.savefig(bio, dpi=250, format="png",
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=True)
    plt.close(plt.gcf())
    plt.clf()
    return Image.open(bio)
    
def setup_plot():
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax

def show_image(ax, img):
    ax.imshow(img)
    return img.width, img.height

def draw_polygon(ax, coords, img_size):
    w, h = img_size
    scaled = [(x * w, y * h) for x, y in coords]
    poly = patches.Polygon(scaled, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(poly)


