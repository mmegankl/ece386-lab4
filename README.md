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
The .tflite model is significantly smaller than the .keras model, allowing for faster loading and execution. This makes the liteRT more efficient for edge computing.
Regarding performance, running 10 images per batch results in smoother runtimes with better performance. Running 40 images per batch increases throughput, hindering performance because it has to process more at once, putting more strain on the memory and causing delays.
LiteRT is overall better than Keras, it was 8 times faster overall and 21 times faster at user processing than Keras. LiteRT runs more efficiently without having to reload memory as often. Keras is slower because of the additional CPU power and memory it needs to run.   
In conclusion, the LiteRT model is the better choice due to its faster run times and efficient memory usage.   
## Documentation

### People

### LLMs
https://box.boodle.ai/c/b5aecd2d-5eb1-4482-b7ca-ada9e0887f13 
