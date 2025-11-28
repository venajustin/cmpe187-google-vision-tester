from io import BytesIO
def localize_objects(img): 

    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    buffer = BytesIO()
    img.save(buffer, img.format)
    image = vision.Image(content=buffer.getvalue())

    objects = client.object_localization(image=image).localized_object_annotations
    return objects
