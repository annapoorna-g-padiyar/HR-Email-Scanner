import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = r"C:\Users\LENOVO\HR_Resume_AI\classified\review\R42.png"

print("=" * 70)
print("OCR TEST")
print("=" * 70)

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found")
    exit()

print("Original image size:", image.shape)

# Resize
image = cv2.resize(
    image,
    None,
    fx=1.5,
    fy=1.5,
    interpolation=cv2.INTER_CUBIC
)

# Grayscale
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# OCR
text = pytesseract.image_to_string(
    gray,
    config="--psm 11"
)

print("\n")
print("=" * 70)
print("OCR OUTPUT")
print("=" * 70)

print(text)

print("=" * 70)
print("OCR STATISTICS")
print("=" * 70)

clean_text = " ".join(text.split())

print("Characters :", len(clean_text))
print("Words      :", len(clean_text.split()))

print("\n")
print("OCR completed.")