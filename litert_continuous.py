"""This script loads a .tflite model into LiteRT and continuously takes pictures with a webcam,
printing if the picture is of a cat or a dog."""

import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys
import numpy as np


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
def resize_pic(img) -> np.ndarray:
    resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(resized_img, (150, 150))  # appropriate size
    resized_img = np.array(resized_img, dtype=np.uint8)  # convert to array
    resized_img = np.expand_dims(resized_img, axis=0)  # matrix
    return resized_img


# TODO: Function to conduct inference
def inference(img: np.ndarray, runner) -> tuple[str, float]:
    input = resize_pic(img)
    # something about prediction, outputting 0 for cat or 1 for dog
    prediction = runner(catdog_input=img)[0][0]
    if prediction <= 0.5:
        class_type = "Cat"
    else:
        class_type = "Dog"

    return class_type, prediction


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

    # TODO: Loop to take pictures and invoke inference. Should loop until Ctrl+C keyboard interrupt.
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("exiting")
    finally:
        # Release the camera
        webcam.release()
        print("Program complete")


# Executes when script is called by name
if __name__ == "__main__":
    main()
