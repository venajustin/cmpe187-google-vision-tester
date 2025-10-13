import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from io import BytesIO
from PIL import Image

def label_img(objects, img, highlight_index = -1):

    fig, ax = setup_plot()
    w, h = show_image(ax, img)

    for index, object_ in enumerate(objects):
        coords = []
        for vertex in object_.bounding_poly.normalized_vertices:
            coords.append((vertex.x,vertex.y))
        if index == highlight_index:
            highlight = True
        else:
            highlight = False
        draw_polygon(ax,coords,(w,h), highlight)

    bio = BytesIO()
    plt.savefig(bio, dpi=250, format="png",
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=True)
    plt.close(plt.gcf())
    plt.close()
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

def draw_polygon(ax, coords, img_size, highlight):
    w, h = img_size
    scaled = [(x * w, y * h) for x, y in coords]
    edgecolor = 'red'
    order = 1
    if highlight:
        edgecolor = 'blue'
        order = 2.5
    poly = patches.Polygon(scaled, fill=False, edgecolor=edgecolor, linewidth=2, zorder=order)
    ax.add_patch(poly)


