import cv2

# Load image
img = cv2.imread('./images/dog.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

# Denoise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Threshold — Otsu automatically finds the best split between warm animal and cool background
_, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphology: close fills holes inside the animal, open removes background noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)

# Find contours, keep largest (the animal)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
animal = max(contours, key=cv2.contourArea)

# Draw boundary on original image
result = img.copy()
cv2.drawContours(result, [animal], -1, (0, 255, 255), 2)   # cyan boundary
x, y, w, h = cv2.boundingRect(animal)
cv2.rectangle(result, (x, y), (x+w, y+h), (0, 80, 255), 2) # bounding box

# Save
cv2.imwrite("output_mask.png", mask)
cv2.imwrite("output_boundary.png", result)
print("Saved: output_mask.png, output_boundary.png")
