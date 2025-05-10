import cv2
import numpy as np
import logging
from pyzbar.pyzbar import decode
import argparse

# Constants
ASPECT_RATIO_MIN = 0.9  # Adjusted from 0.8
ASPECT_RATIO_MAX = 1.1  # Adjusted from 1.2
AREA_RATIO_MIN = 0.04   # Adjusted from 0.05
POSITION_MARKER_AREA_MIN = 1000
POSITION_MARKER_AREA_MAX = 10000
ALIGNMENT_MARKER_AREA_MIN = 500
ALIGNMENT_MARKER_AREA_MAX = 2000

logging.basicConfig(level=logging.INFO)

def is_square(approx):
    """Check if a contour is a square based on aspect ratio and area."""
    (x, y, w, h) = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(approx)
    perimeter = cv2.arcLength(approx, True)
    return (ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX and
            (area / (perimeter ** 2)) > AREA_RATIO_MIN)

def is_less_than_half_black(img, contour):
    """Check if less than half of the contour area is black."""
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y+h, x:x+w]
    total_pixels = roi.size
    black_pixels = np.sum(roi == 0)
    return black_pixels <= 0.6 * total_pixels

def is_inside_position_marker(contour, position_markers):
    """Check if a contour is inside any of the position markers."""
    for position_marker in position_markers:
        for point in contour:
            if cv2.pointPolygonTest(position_marker, (int(point[0][0]), int(point[0][1])), False) >= 0:
                return True
    return False

def filter_contours(contours, img):
    position_markers = []
    alignment_markers = []

    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        if len(approx) >= 4 and is_square(approx):  # Allow more flexibility in the number of vertices
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if (area / (perimeter ** 2)) > AREA_RATIO_MIN:
                if POSITION_MARKER_AREA_MIN < area < POSITION_MARKER_AREA_MAX:
                    if is_less_than_half_black(img, contour):
                        position_markers.append(contour)
                elif ALIGNMENT_MARKER_AREA_MIN < area < ALIGNMENT_MARKER_AREA_MAX:
                    if not is_inside_position_marker(contour, position_markers) and is_less_than_half_black(img, contour):
                        alignment_markers.append(contour)

    return position_markers, alignment_markers

def find_markers(img):
    """Find and filter position and alignment markers in the image."""
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    position_markers, alignment_markers = filter_contours(contours, img)
    verified_alignment_markers = verify_alignment_markers(alignment_markers, position_markers)
    return position_markers, verified_alignment_markers

def verify_alignment_markers(alignment_markers, position_markers):
    """Verify alignment markers are not inside position markers."""
    verified_alignment_markers = []
    for alignment_marker in alignment_markers:
        if not is_inside_position_marker(alignment_marker, position_markers):
            verified_alignment_markers.append(alignment_marker)
    return verified_alignment_markers

def preprocess_image(image_path):
    """Preprocess the image for better marker detection."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logging.error("Erro ao carregar a imagem.")
        return None

    resized_image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    adaptive_thresh = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 211, 1)
    equalized_image = cv2.equalizeHist(adaptive_thresh)
    filtered_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)
    sharpened_image = sharpen_image(filtered_image)

    return sharpened_image

def sharpen_image(img):
    """Apply sharpening filter to the image."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def calculate_inner_square_thickness(alignment_markers):
    """Calculate the thickness of the inner square of alignment markers."""
    if not alignment_markers:
        return 0
    inner_thicknesses = [cv2.boundingRect(marker)[2] // 3 for marker in alignment_markers]
    return int(np.mean(inner_thicknesses))

def draw_red_grid(img, marker_thickness, offset_x=0, offset_y=0):
    """Draw a red grid on the image."""
    img_height, img_width = img.shape[:2]
    for x in range(offset_x, img_width, marker_thickness):
        cv2.line(img, (x, 0), (x, img_height), (0, 0, 255), 1)
    for y in range(offset_y, img_height, marker_thickness):
        cv2.line(img, (0, y), (img_width, y), (0, 0, 255), 1)
    return img

def count_grid_lines_and_columns(img, marker_thickness, offset_x=0, offset_y=0):
    """Count the number of grid lines and columns."""
    img_height, img_width = img.shape[:2]
    num_columns = sum(np.any(img[:, x:x+marker_thickness] == 0) for x in range(offset_x, img_width, marker_thickness))
    num_rows = sum(np.any(img[y:y+marker_thickness, :] == 0) for y in range(offset_y, img_height, marker_thickness))
    return num_rows, num_columns

def get_qr_code_version_by_lines(num_lines):
    """Get QR code version based on the number of lines."""
    version_mapping = {
        21: 1, 25: 2, 29: 3, 33: 4, 37: 5, 41: 6, 45: 7, 49: 8, 53: 9, 57: 10,
        61: 11, 65: 12, 69: 13, 73: 14, 77: 15, 81: 16, 85: 17, 89: 18, 93: 19, 97: 20,
        101: 21, 105: 22, 109: 23, 113: 24, 117: 25, 121: 26, 125: 27, 129: 28, 133: 29, 137: 30,
        141: 31, 145: 32, 149: 33, 153: 34, 157: 35, 161: 36, 165: 37, 169: 38, 173: 39, 177: 40
    }
    return version_mapping.get(num_lines, "versão desconhecida")

def get_qr_code_version_by_alignment(num_alignment_markers):
    """Get QR code version based on the number of alignment markers."""
    version_mapping = {
        0: [1], 1: [2, 3, 4, 5, 6], 6: [7, 8, 9, 10, 11, 12, 13], 13: [14, 15, 16, 17, 18, 19, 20],
        22: [21, 22, 23, 24, 25, 26, 27], 33: [28, 29, 30, 31, 32, 33], 46: [34, 35, 36, 37, 38, 39, 40]
    }
    return version_mapping.get(num_alignment_markers, ["versão desconhecida"])

def decode_qr_code(image):
    """Decode QR code from the image."""
    decoded_objects = decode(image)
    return [obj.data.decode('utf-8') for obj in decoded_objects]

def is_center_black(img, contour):
    """Check if the center of the contour is black."""
    x, y, w, h = cv2.boundingRect(contour)
    center_x, center_y = x + w // 2, y + h // 2
    margin = 2
    roi = img[center_y - margin:center_y + margin, center_x - margin:center_x + margin]
    return np.all(roi == 0)

def replace_center_with_black(img, contour):
    """Replace the center of the contour with black."""
    x, y, w, h = cv2.boundingRect(contour)
    center_x, center_y = x + w // 2, y + h // 2
    margin_w, margin_h = w // 5, h // 5
    img[center_y - margin_h:center_y + margin_h, center_x - margin_w:center_x + margin_w] = 0

def painted_yellow_lines(img, position_markers, alignment_markers):
    """Draw yellow lines connecting markers in the same row or column."""
    all_markers = position_markers + alignment_markers
    marker_centers = []
    for marker in all_markers:
        x, y, w, h = cv2.boundingRect(marker)
        center_x = x + w // 2
        center_y = y + h // 2
        marker_centers.append((center_x, center_y))

    marker_centers.sort(key=lambda point: (point[1], point[0]))
    draw_lines(img, marker_centers, axis='horizontal')

    marker_centers.sort(key=lambda point: (point[0], point[1]))
    draw_lines(img, marker_centers, axis='vertical')

def draw_lines(img, points, axis):
    """Draw lines on the image based on the axis."""
    if axis == 'horizontal':
        for i in range(1, len(points)):
            if points[i][1] == points[i - 1][1]:
                cv2.line(img, points[i - 1], points[i], (0, 255, 255), 2)
    elif axis == 'vertical':
        for i in range(1, len(points)):
            if points[i][0] == points[i - 1][0]:
                cv2.line(img, points[i - 1], points[i], (0, 255, 255), 2)

def main():
    """Main function to process the image and detect QR code markers."""
    parser = argparse.ArgumentParser(description="Processa uma imagem para detectar marcadores de QR Code.")
    parser.add_argument("image_path", type=str, help="Caminho para a imagem a ser processada.")
    args = parser.parse_args()

    image_path = args.image_path

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Erro ao carregar a imagem: {image_path}")
            return

        position_markers, alignment_markers = find_markers(img)
        marker_thickness = calculate_inner_square_thickness(alignment_markers)

        if marker_thickness == 0:
            img = preprocess_image(image_path)
            position_markers, alignment_markers = find_markers(img)
            if not position_markers and not alignment_markers:
                logging.info("No alignment markers detected to calculate marker thickness.")
                return
            marker_thickness = calculate_inner_square_thickness(alignment_markers)

        if alignment_markers:
            offset_x = cv2.boundingRect(alignment_markers[0])[0] % marker_thickness
            offset_y = cv2.boundingRect(alignment_markers[0])[1] % marker_thickness
        else:
            logging.info("No alignment markers detected.")
            return

        for marker in position_markers + alignment_markers:
            if not is_center_black(img, marker):
                replace_center_with_black(img, marker)

        output_path = 'fixed_qr.png'
        cv2.imwrite(output_path, img)
        logging.info(f"QR fixed em: {output_path}")

        if len(img.shape) == 2 or img.shape[2] == 1:
            img_with_grid = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_with_grid = img

        logging.info(f"Decoded QR code: {decode_qr_code(img_with_grid)}")

        for marker in position_markers:
            x, y, w, h = cv2.boundingRect(marker)
            cv2.rectangle(img_with_grid, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img_with_grid, (x + marker_thickness, y + marker_thickness),
                          (x + w - marker_thickness, y + h - marker_thickness), (0, 255, 0), 2)
            cv2.putText(img_with_grid, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        for marker in alignment_markers:
            x, y, w, h = cv2.boundingRect(marker)
            cv2.rectangle(img_with_grid, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(img_with_grid, (x + marker_thickness, y + marker_thickness),
                          (x + w - marker_thickness, y + h - marker_thickness), (255, 0, 0), 2)
            cv2.putText(img_with_grid, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        img_with_grid = draw_red_grid(img_with_grid, marker_thickness, offset_x, offset_y)
        painted_yellow_lines(img_with_grid, position_markers, alignment_markers)

        num_rows, num_columns = count_grid_lines_and_columns(img, marker_thickness, offset_x, offset_y)
        qr_code_version_by_lines = get_qr_code_version_by_lines(num_rows)
        qr_code_version_by_alignment = get_qr_code_version_by_alignment(len(alignment_markers))

        logging.info(f"colunas: {num_columns}")
        logging.info(f"linhas: {num_rows}")
        logging.info(f"pos marcador: {len(position_markers)}")
        logging.info(f"alin marcador: {len(alignment_markers)}")
        logging.info(f"QR code versão: {qr_code_version_by_lines}")
        logging.info(f"QR code possíveis versões: {qr_code_version_by_alignment}")

        output_path = 'detected_markers_grid.png'
        cv2.imwrite(output_path, img_with_grid)
        logging.info(f"Grid salvo em: {output_path}")

    except Exception as e:
        logging.error(f"erro: {e}")

if __name__ == "__main__":
    main()
