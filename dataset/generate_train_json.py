import os

image_dir = "non-hair"
conditioning_image_dir = "hair"


with open("metadata.json", "w") as f:
    for filename in sorted(os.listdir(image_dir)):
        if not filename.endswith(".png"):
            continue
        txt = "{"
        txt += '"text": ""'
        txt += ", "
        txt += f'"image": "{image_dir}/{filename}"'
        txt += ", "
        txt += f'"conditioning_image": "{conditioning_image_dir}/{filename}"'
        txt += "}\n"
        f.write(txt)
