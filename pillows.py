from PIL import Image
img = Image.open("gs1.jpg")
img.show()
print(img.mode)