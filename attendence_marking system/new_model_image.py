import cv2
import face_recognition
import numpy as np
import os
import time
from datetime import datetime
import csv

def load_rfid_name_mapping(csv_file="rfid_to_name.csv"):
    rfid_to_name = {}
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rfid_to_name[row['RFID']] = row['Name']
        print(f"Loaded {len(rfid_to_name)} RFID-to-name mappings")
    except FileNotFoundError:
        print(f"Warning: RFID mapping file {csv_file} not found")
    except Exception as e:
        print(f"Error loading RFID mapping: {e}")
    return rfid_to_name

def load_known_faces(image_dir="captured_images", rfid_to_name=None):
    known_face_encodings = []
    known_face_names = []
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        try:
            rfid = image_file.split('_')[0]
            name = rfid_to_name.get(rfid, rfid) if rfid_to_name else rfid
            
            image_path = os.path.join(image_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"Loaded face for: {name} (RFID: {rfid})")
            else:
                print(f"No faces found in image: {image_file}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return known_face_encodings, known_face_names

def run_real_time_detection(known_face_encodings, known_face_names):
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    #trying to set some check points 
    checkpoint_times = {name: ["Not Detected", "Not Detected", "Not Detected"] for name in known_face_names}
    checkpoint_count = 0
    last_checkpoint_time = time.time()
    checkpoint_interval = 20  #  setting up 10 seconds
    
    print("Starting periodic attendance check (3 checkpoints every 10 seconds)...")
    
    while checkpoint_count < 2:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Processing  all the  frame
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            current_time = time.time()
            
            # Check if it's time for a new checkpoint or not
            if current_time - last_checkpoint_time >= checkpoint_interval:
                checkpoint_count += 1
                last_checkpoint_time = current_time
                print(f"\nStarting checkpoint {checkpoint_count}/3 at {datetime.now().strftime('%H:%M:%S')}")
                
                detected_in_checkpoint = set()
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        accuracy = (1 - face_distances[best_match_index]) * 100
                        
                        if accuracy > 50 and name not in detected_in_checkpoint:
                            checkpoint_times[name][checkpoint_count-1] = datetime.now().strftime('%H:%M:%S')
                            detected_in_checkpoint.add(name)
                            print(f"  Marked present: {name}")
                
                # For those not detected in this checkpoint
                for name in known_face_names:
                    if name not in detected_in_checkpoint:
                        checkpoint_times[name][checkpoint_count-1] = "Absent"
            
            # Regular face detection (for display only)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Display the results results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Display checkpoint information
        y_offset = 30
        cv2.putText(frame, f"Checkpoint {checkpoint_count}/3", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        time_until_next = max(0, checkpoint_interval - (time.time() - last_checkpoint_time))
        cv2.putText(frame, f"Next in: {time_until_next:.1f}s", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        # Displaying the current check point status
        for name in known_face_names:
            status = checkpoint_times[name][checkpoint_count-1] if checkpoint_count > 0 else "Pending"
            color = (0, 255, 0) if status not in ["Not Detected", "Absent"] else (0, 0, 255)
            text = f"{name}: {status}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        cv2.imshow('Periodic Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # calculating final attendence
    attendance_records = {}
    for name in known_face_names:
        present_count = sum(1 for status in checkpoint_times[name] if status not in ["Not Detected", "Absent"])
        final_status = "Present" if present_count >= 2 else "Absent"  # Present if detected in 2+ checkpoints
        attendance_records[name] = (*checkpoint_times[name], final_status)
    
    save_attendance(attendance_records)
    video_capture.release()
    cv2.destroyAllWindows()

def save_attendance(attendance_records):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    
    with open('periodic_attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3', 'Status'])
        for name, (cp1, cp2, cp3, status) in attendance_records.items():
            writer.writerow([name, date, cp1, cp2, cp3, status])
    print("Periodic attendance saved to periodic_attendance.csv")

if __name__ == "__main__":
    rfid_to_name = load_rfid_name_mapping()
    known_face_encodings, known_face_names = load_known_faces(rfid_to_name=rfid_to_name)
    
    if not known_face_encodings:
        print("No faces found in captured_images directory. Exiting.")
    else:
        print(f"Loaded {len(known_face_names)} known faces")
        run_real_time_detection(known_face_encodings, known_face_names)