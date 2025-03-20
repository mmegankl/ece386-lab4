# LiteRT Inference with Webcam

## Usage

Tested on Raspberry Pi 5 with USB Webcam.

After cloning,

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements-lite.txt`
4. Copy in your `.tflite` model from [Prelab](https://usafa-ece.github.io/ece386-book/b3-devboard/lab-cat-dog.html#pre-lab)
5. `python litert_continuous.py cat-dog-mnv2.tflite`

Verify your signatures are what you expect, then get to work!

## Discussion Questions
Compare

    Running your .tflite model with 10 images per run vs 40 images per run

    Running your .tflite model with LiteRT vs your .keras model with Keras  

Discuss the results in your README.md. Make sure you emphasize

    Relative model sizes

    Relative performance for more vs. fewer images per run, and why

    Pipeline stalls waiting for memory

    L2 invalidations (meaning something in the L2 cache had to be overwritten)

    LLC loads and misses
## Documentation

### People

### LLMs
https://box.boodle.ai/c/b5aecd2d-5eb1-4482-b7ca-ada9e0887f13 
