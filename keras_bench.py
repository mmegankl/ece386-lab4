"""This is a benchmark for running Keras model.
Try to make this code as similar to litert_benchmark as possible,
for a fair comparison."""

import cv2
import time
import sys
import numpy as np
import os  # For environment variable checking
import tensorflow as tf  # For loading and running Keras models


def load_keras_model(model_path: str) -> tf.keras.Model:
    """
    Loads a Keras model from the specified path

    Args:
        model_path (str): Path to a .keras or .h5 model file

    Returns:
        tf.keras.Model: Loaded Keras model ready for inference
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded Keras model: {model.summary()}")

    # Warm up the model (first inference is often slower)
    warmup_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
    _ = model.predict(warmup_input)

    return model


def resize_pic_keras(img) -> np.ndarray:
    """
    Resize and preprocess an image for Keras model input

    Args:
        img: Input image in BGR format (OpenCV default)

    Returns:
        np.ndarray: Preprocessed image ready for model inference
    """
    # Convert from BGR (OpenCV format) to RGB (model expects RGB)
    resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to the dimensions expected by the model (150x150)
    resized_img = cv2.resize(resized_img, (150, 150))

    # Convert to float32 and normalize to [0,1] range for Keras
    resized_img = np.array(resized_img, dtype=np.float32) / 255.0

    # Add batch dimension
    resized_img = np.expand_dims(resized_img, axis=0)

    return resized_img


def inference_keras(img: np.ndarray, model) -> tuple[str, float]:
    """
    Perform inference on an image using the provided Keras model

    Args:
        img: Input image
        model: Keras model

    Returns:
        tuple[str, float]: Class prediction (Cat/Dog) and confidence value
    """
    # Preprocess the image for the model
    input_data = resize_pic_keras(img)

    # Run inference with the Keras model
    prediction = model.predict(input_data, verbose=0)[0][0]

    # Interpret the prediction value
    if prediction <= 0.5:
        class_type = "Cat"
    else:
        class_type = "Dog"

    return class_type, float(prediction)


def main():
    # Check if we're running in headless mode for perf testing
    headless_mode = os.environ.get("HEADLESS", "false").lower() == "true"

    # Verify command line arguments
    if len(sys.argv) != 2:
        print("Usage: python keras_benchmark.py <model_path.keras>")
        exit(1)

    # Load Keras model from path given as argument
    model_path = sys.argv[1]
    model = load_keras_model(model_path)

    # Initialize webcam
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

            # Measure inference time
            start_time = time.time()

            # Run inference
            class_type, confidence = inference_keras(frame, model)

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
                cv2.imshow("Pet Detector Benchmark (Keras)", display_frame)

                # Wait for a short time between frames (10ms)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    print("Benchmark interrupted by user")
                    break

        # Calculate and display benchmark results
        if successful_inferences > 0:
            avg_inference_time = total_inference_time / successful_inferences
            print(f"\nBenchmark Results for Keras model:")
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
