# END TO END TEXT RECOGNIZATION AND SUMMARIZATION SYSTEM

## OCR SYSTEM (TEXT DETECTION & TEXT RECOGNIZATION)

There are 2 subsystems inside OCR System including: Text detection & Text recognition

### TEXT DETECTION PIPELINE (Using PaddleOCR)

1. Preprocess for text detection model 
- Input: image

2. Infer model
-Input: output of text detection preprocess model

3. Postprocess for text detection model
-Input: output of text detection inference model

4. Serving model using TRITON SERVER
- Preprocess: Triton inference server - python backend
- Inference: Triton inference server - onnxruntime (or tensorrt)
- Postprocess: Triton inference server - python backend

NOTE: Output text detection model is bounding boxes (shape: batch_size, -1, 4, 2)

### FACE RECOGNITION PIPELINE (Using PaddleOCR)

1. Preprocess for text recognition model 
- Input: image, output of text detection postprocess model

2. Infer model
- Input: output of text recognition preprocess model

3. Postprocess for text recognition model
- Input: output of text recognition postprocess model

4. Serving model using TRITON SERVER
- Preprocess: Triton inference server - python backend
- Inference: Triton inference server - onnxruntime (or tensorrt)
- Postprocess: Triton inference server - python backend

NOTE: Output text recognition model is text, score 


## SUMMARIZATION SYSTEM 

1. Data transformation: convert examples to features

2. Data validation: Check whether the dataset file is in the correct form or not

3. Model Trainer: Using pretrained model to train the dataset

4. Prediction: With trained model applying it to predict on the custom paragraphs

NOTE: Input of the text recognition will be the output of Summarization System


## PROCESSES OF AUTO RECOGNIZED AND SUMMARIZED SYSTEM

TEXT DETECTION -> TEXT RECOGNITION -> TEXT SUMMARIZATION

1. Gain output of OCR SYSTEM (text + score)

2. Push it onto MONGO DATABASE to store the value

3. Pull the stored values in MONGO DATABASE to get the predicted summarization using SUMMARIZATION SYSTEM

4. Deploy it by using FASTAPI


