from os import listdir

def get_files(path):
    try: 
        files = listdir(path)
    except Exception:
        print("Ensure ./input folder exists and contains input images")
        return []

    return files
