import cv2
import serial
import os
import time
from datetime import datetime
import argparse
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)  # Higher resolution for better quality
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--serial_port", type=str, default="COM3")
    parser.add_argument("--image_dir", type=str, default="captured_images")
    parser.add_argument("--rfid_csv", type=str, default="rfid_logs.csv")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay in seconds before capturing image")
    parser.add_argument("--box_size", type=float, default=0.6, help="Size of capture box relative to frame (0.1-0.9)")
    parser.add_argument("--output_size", type=int, default=1080, help="Output image size (will be square)")
    return parser.parse_args()

def initialize_serial(port):
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        print(f"Connected to ESP32 on {port}")
        return ser
    except Exception as e:
        print(f"Error connecting to ESP32: {e}")
        return None

def read_rfid(ser):
    try:
        if ser and ser.in_waiting > 0:
            raw_data = ser.readline()
            rfid_uid = raw_data.decode('utf-8', errors='replace').strip()
            rfid_uid = ''.join(c if c.isprintable() else '' for c in rfid_uid)
            if rfid_uid and rfid_uid != "RFID Reader Initialized" and rfid_uid != "Scan an RFID tag to see its UID...":
                return rfid_uid
        return None
    except Exception as e:
        print(f"Error reading RFID: {e}")
        return None

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def initialize_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['RFID', 'Timestamp', 'Image_Path'])

def log_rfid(csv_path, rfid_uid, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([rfid_uid, timestamp, image_path])
    print(f"Logged RFID {rfid_uid} to {csv_path}")

def save_image(image, rfid_uid, image_dir, box_coords, output_size=1080):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{image_dir}/{rfid_uid}_{timestamp}.jpg"
    
    # Crop to the box area
    x, y, w, h = box_coords
    cropped_image = image[y:y+h, x:x+w]
    
    # Resize to 1080x1080 while maintaining aspect ratio
    # First pad if needed to make square
    height, width = cropped_image.shape[:2]
    if width != height:
        size = max(width, height)
        pad_x = (size - width) // 2
        pad_y = (size - height) // 2
        padded_image = cv2.copyMakeBorder(cropped_image, 
                                         pad_y, pad_y, 
                                         pad_x, pad_x, 
                                         cv2.BORDER_CONSTANT, 
                                         value=[0, 0, 0])
    else:
        padded_image = cropped_image
    
    # Resize to target output size
    resized_image = cv2.resize(padded_image, (output_size, output_size))
    
    cv2.imwrite(filename, resized_image)
    print(f"Saved 1080x1080 image: {filename}")
    return filename

def draw_capture_box(image, box_coords, color=(0, 255, 0), thickness=2):
    x, y, w, h = box_coords
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    
    # Draw crosshair in the center
    center_x, center_y = x + w//2, y + h//2
    cv2.line(image, (center_x-20, center_y), (center_x+20, center_y), color, thickness)
    cv2.line(image, (center_x, center_y-20), (center_x, center_y+20), color, thickness)
    
    return image

def main():
    args = get_args()
    
    # Validate box size
    args.box_size = max(0.1, min(0.9, args.box_size))
    
    # Create directories and initialize CSV
    ensure_directory_exists(args.image_dir)
    initialize_csv(args.rfid_csv)
    
    # Initialize camera with higher resolution
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Could not open camera with device index {args.device}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Initialize serial connection
    esp32_serial = initialize_serial(args.serial_port)
    if not esp32_serial:
        print("Failed to connect to ESP32. Exiting.")
        return
    
    print("Instructions:")
    print(f"- Position your face within the green frame")
    print(f"- Scan an RFID tag to capture an image after {args.delay} seconds")
    print(f"- Images will be saved as {args.output_size}x{args.output_size} pixels")
    print("- Press 'q' to quit")
    
    rfid_uid = None
    capture_time = None
    counting_down = False

    while True:
        # Check for RFID scan if not already counting down
        if not counting_down:
            rfid_uid = read_rfid(esp32_serial)
            if rfid_uid:
                print(f"RFID scanned: {rfid_uid}")
                print(f"Capturing image in {args.delay} seconds...")
                capture_time = time.time() + args.delay
                counting_down = True
        
        # Capture camera frame
        ret, image = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Flip and display
        image = cv2.flip(image, 1)
        debug_image = image.copy()
        
        # Calculate box coordinates (centered)
        height, width = image.shape[:2]
        box_width = int(width * args.box_size)
        box_height = int(height * args.box_size)
        
        # Make the box square based on the smaller dimension
        box_size = min(box_width, box_height)
        box_x = (width - box_size) // 2
        box_y = (height - box_size) // 2
        box_coords = (box_x, box_y, box_size, box_size)
        
        if counting_down:
            # Show countdown timer
            remaining = max(0, capture_time - time.time())
            if remaining > 0:
                countdown_text = f"Capturing in: {remaining:.1f}s"
                cv2.putText(debug_image, countdown_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(debug_image, f"RFID: {rfid_uid}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Flash the box red during countdown
                box_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 255, 0)
                debug_image = draw_capture_box(debug_image, box_coords, box_color, 3)
            else:
                # Time's up - capture the image and log to CSV
                image_path = save_image(image, rfid_uid, args.image_dir, box_coords, args.output_size)
                log_rfid(args.rfid_csv, rfid_uid, image_path)
                counting_down = False
                rfid_uid = None
                cv2.putText(debug_image, "Image captured!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Flash green box to confirm capture
                debug_image = draw_capture_box(debug_image, box_coords, (0, 255, 0), 4)
        else:
            # Ready for next scan - show static green box
            debug_image = draw_capture_box(debug_image, box_coords)
            cv2.putText(debug_image, "Ready to scan", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(debug_image, "Position face within frame", (width//2-150, box_y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('RFID Image Capture', debug_image)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if esp32_serial:
        esp32_serial.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()