import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load images and get face encodings
known_faces = {
    "jobs": face_recognition.face_encodings(face_recognition.load_image_file("photos/jobs.jpg"))[0],
    "shiva": face_recognition.face_encodings(face_recognition.load_image_file("photos/shiva.jpg"))[0],
 
}

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
frame_count = 0

# Get current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file for writing
with open(current_date + '.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    # Write the header row in the CSV file
    lnwriter.writerow(["Name", "Time"])

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Process every 5th frame to reduce CPU load
        if frame_count % 5 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

                if name in known_face_names and name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    print(f"{name} marked present at {current_time}")

                    # Display the name on the video frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 100)
                    fontScale = 1.5
                    fontColor = (255, 0, 0)
                    thickness = 3
                    lineType = 2

                    cv2.putText(frame, name + ' Present',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

        # Draw rectangles around the faces
        for face_location in face_locations:
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


        # Display the resulting frame
        cv2.imshow('Attendance System', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# Release the capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
