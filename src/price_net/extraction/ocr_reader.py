import easyocr

# Create a reader object
reader = easyocr.Reader(['en'])

# Read text from an image
results = reader.readtext('/Users/porterjenkins/data/price-attribution-scenes/test/price-images/0a412b0c-ec3e-4ce9-bcb8-7ea7c8b503f1.jpg', detail=0)
results_str = ", ".join(results)
print(results)