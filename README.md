# END TO END TEXT RECOGNIZATION AND SUMMARIZATION SYSTEM

## OCR SYSTEM (TEXT DETECTION & TEXT RECOGNIZATION)

### TEXT DETECTION PIPELINE (Using PaddleOCR)

1. Preprocess for text detection model 
- Input: image

2. Infer model
-Input: output of text detection preprocess model

3. Postprocess for text detection model
-Input: output of text detection inference model

4. Serving model using Triton Server
- Preprocess: Triton inference server - python backend
- Inference: Triton inference server - onnxruntime
- Postprocess: Triton inference server - python backend

### FACE RECOGNITION PIPELINE (Using PaddleOCR)

1. Preprocess for text recognition model 
- Input: image, output of text detection postprocess model

2. Infer model
- Input: output of text recognition preprocess model

3. Postprocess for text recognition model
- Input: output of text recognition postprocess model

4. Serving model using Triton Server
- Preprocess: Triton inference server - python backend
- Inference: Triton inference server - onnxruntime
- Postprocess: Triton inference server - python backend

## SUMMARIZATION SYSTEM 


