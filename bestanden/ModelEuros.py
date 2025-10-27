import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Load the labels
with open("labels.txt", "r") as file:
    class_names = file.read().splitlines()

# Create the array of the right shape to feed into the TensorFlow Lite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Resize the frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Normalize the image
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    
    # Set input tensor
    input_data = normalized_frame[np.newaxis, ...]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Perform the inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get prediction
    prediction = np.argmax(output_data)
    class_name = class_names[prediction]
    confidence_score = output_data[0][prediction]
    
    # Display prediction and confidence score on the frame
    cv2.putText(frame, f"{class_name}: {confidence_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
