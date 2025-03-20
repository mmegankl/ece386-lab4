"""This script loads a .tflite model into LiteRT and continuously takes pictures with a webcam,
printing if the picture is of a cat or a dog."""

import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def get_litert_runner(model_path: str) -> SignatureRunner:
    """Opens a .tflite model from path and returns a LiteRT SignatureRunner that can be called for inference

    Args:
        model_path (str): Path to a .tflite model

    Returns:
        SignatureRunner: An AI-Edge LiteRT runner that can be invoked for inference."""

    interpreter = Interpreter(model_path=model_path)
    # Allocate the model in memory. Should always be called before doing inference
    interpreter.allocate_tensors()
    print(f"Allocated LiteRT with signatures {interpreter.get_signature_list()}")

    # Create callable object that runs inference based on signatures
    # 'serving_default' is default... but in production should parse from signature
    return interpreter.get_signature_runner("serving_default")


# TODO: Function to resize picture and then convert picture to numpy for model ingest
# C1C Harkley helped me get this started and helped me to understand this part (the different elements and why)
# good
def resize_pic(img) -> np.ndarray:
    resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(resized_img, (150, 150))  # appropriate size
    resized_img = np.array(resized_img, dtype=np.uint8)  # convert to array
    resized_img = np.expand_dims(resized_img, axis=0)  # matrix
    return resized_img


# TODO: Function to conduct inference
def inference(img: np.ndarray, runner) -> tuple[str, float]:
    input_data = resize_pic(img)

    # Call the runner with the resized input
    result = runner(catdog_input=input_data)

    # Get the output tensor
    prediction_array = result["output_0"]

    # Extract the first value from the array
    prediction_value = prediction_array[0, 0]

    # Now we can safely compare a single value
    if prediction_value <= 0.5:
        class_type = "Cat"
    else:
        class_type = "Dog"

    return class_type, prediction_value


def main():

    # Verify arguments
    if len(sys.argv) != 2:
        print("Usage: python litert.py <model_path.tflite>")
        exit(1)

    # Create LiteRT SignatureRunner from model path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)

    # Print input and output details of runner
    print(f"Input details:\n{runner.get_input_details()}")
    print(f"Output details:\n{runner.get_output_details()}")

    # Init webcam
    webcam = cv2.VideoCapture(0)  # 0 is default camera index
    try:
        while True:
            ret, frame = webcam.read()
            if not ret:  # Only process of ret is True
                print("Failed to capture image")
                break

            # Make a copy for display
            display_frame = frame.copy()

            # Prepare input for the model
            model_input = resize_pic(frame)
            print("Image shape:", model_input.shape)

            # Run inference
            class_type, confidence = inference(frame, runner)

            # Display result on frame
            cv2.putText(
                display_frame,
                f"{class_type}: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show the original frame (not the resized/processed one)
            cv2.imshow("Pet Detector", display_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt")
    finally:
        # Release resources
        webcam.release()
        cv2.destroyAllWindows()
        print("Program complete")
    # TODO: Loop to take pictures and invoke inference. Should loop until Ctrl+C keyboard interrupt.
    # try:
    #     while True:
    #         ret, frame = webcam.read()
    #         class_type, confidence = inference(frame, runner)
    #         cv2.imshow("Captured Image", frame)
    #         print(f"Prediction: {class_type} with {confidence} certainty")
    # except KeyboardInterrupt:
    #     print("exiting")
    # finally:
    #     # Release the camera
    #     webcam.release()
    #     print("Program complete")
    # Capture a frame
    # ret, frame = webcam.read()

    # Release the camera
    # webcam.release()


"""
    # Only process of ret is True
    if ret:
        # Convert BGR (OpenCV default) to RGB for TFLite
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to a NumPy array
        # img_array = np.array(frame_rgb, dtype=np.uint8)
        new_frame = resize_pic(frame)
        class_type, confidence = inference(frame, runner)
        print("Image shape:", new_frame.shape)  # Ensure shape matches model input

        # Preview the image
        cv2.imshow("Captured Image", new_frame)
        print(f"Prediction: {class_type} with {confidence} certainty")
        print("Press any key to exit.")
        while True:
            # Window stays open until key press
            if cv2.waitKey(0):
                cv2.destroyAllWindows()
                break

    else:
        print("Failed to capture image.")
"""

# Executes when script is called by name
if __name__ == "__main__":
    main()
