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
Regarding performance, running 10 images per batch results in smoother runtimes with better performance. Running 40 images per batch increases throughput, hindering performance because it has to process more at once, putting more strain on the memory and causing delays. Further, pipeline stalls should be more prevalent in the Keras model because it requires more memory. The L2 cache stores frequently-accessed data, so when L2 invalidation occurs (data that is still relevant is overwritten), the data needs to be reloaded, causing pipeline delays. LLCs are loaded when the system finds the other level caches to be preoccupied, and further, a miss happends when the data is not in the LLC either, so it has to fetch the data from the RAM, which is less efficient. Keras is more prone to these issues.
LiteRT is overall better than Keras, it was 8 times faster overall and 21 times faster at user processing than Keras (reference Capt Yarbrough's performance table). LiteRT runs more efficiently without having to reload memory as often. Keras is slower because of the additional CPU power and memory it needs to run.   
In conclusion, the LiteRT model is the better choice due to its faster run times and efficient memory usage.   

## Documentation
C2C Leong got Ei from Capt Yarbrough for help with the continuous file. We used GeeksForGeeks/StackOverflow to help with some code. Ai was used in a learning manner, and not to get the answer.

### People
Capt Yarbrough

### LLMs
https://box.boodle.ai/c/b5aecd2d-5eb1-4482-b7ca-ada9e0887f13 
