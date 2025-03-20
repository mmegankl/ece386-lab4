"""This is a benchmark for running LiteRT model.
After completing the litert_continuous.py
simply copy in the code here and replace the while loop with a for loop"""

import cv2
import time
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import os


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
    # From chat, not too sure
    # Check if we're running in headless mode for perf testing
    # Set this environment variable when running with perf
    headless_mode = os.environ.get("HEADLESS", "false").lower() == "true"

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

    # Number of images to process - can be controlled via environment variable
    num_images = int(os.environ.get("NUM_IMAGES", "10"))
    print(f"Processing {num_images} images")

    # Variables for benchmarking
    total_inference_time = 0
    successful_inferences = 0

    try:
        # Process exactly the specified number of images
        for i in range(num_images):
            # Capture a frame from the webcam
            ret, frame = webcam.read()
            if not ret:
                print(f"Failed to capture image {i+1}")
                continue

            # Prepare input for the model
            model_input = resize_pic(frame)

            # Measure inference time
            start_time = time.time()

            # Run inference
            class_type, confidence = inference(frame, runner)

            # Calculate inference time
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            successful_inferences += 1

            # Print result for this image
            print(
                f"Image {i+1}: {class_type} with confidence {confidence:.4f}, inference time: {inference_time:.4f}s"
            )

            # Only display if not in headless mode
            if not headless_mode:
                # Make a copy for display
                display_frame = frame.copy()

                # Display result on frame
                cv2.putText(
                    display_frame,
                    f"{class_type}: {confidence:.2f} ({inference_time:.4f}s)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Show the original frame with prediction overlay
                cv2.imshow("Pet Detector Benchmark", display_frame)

                # Wait for a short time between frames (10ms)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    print("Benchmark interrupted by user")
                    break

        # Calculate and display benchmark results
        if successful_inferences > 0:
            avg_inference_time = total_inference_time / successful_inferences
            print(f"\nBenchmark Results for .tflite model with LiteRT:")
            print(f"Total images processed: {successful_inferences}")
            print(f"Total inference time: {total_inference_time:.4f} seconds")
            print(f"Average inference time per image: {avg_inference_time:.4f} seconds")
            print(f"Throughput: {1/avg_inference_time:.2f} images per second")

    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        # Release resources
        webcam.release()
        if not headless_mode:
            cv2.destroyAllWindows()
        print("Benchmark complete")


if __name__ == "__main__":
    main()
