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

![triton_det](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/3f297055-78f6-4167-b0cd-354203922c84)

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

![triton_rec](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/e462aca4-365e-41d1-8b77-31766675fa04)

## SUMMARIZATION SYSTEM 

1. Data transformation: convert examples to features (Using Samsum dataset at HuggingFace)

2. Data validation: Check whether the dataset file is in the correct form or not

3. Model Trainer: Using pretrained model to train the dataset (Using pretrained Google/Pegasus model)

4. Prediction: With trained model applying it to predict on the custom paragraphs

NOTE: Input of the text recognition will be the output of Summarization System


## PROCESSES OF AUTO RECOGNIZED AND SUMMARIZED SYSTEM

TEXT DETECTION -> TEXT RECOGNITION -> TEXT SUMMARIZATION

1. Gain output of OCR SYSTEM (text + score)

![text_recognition](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/3ceaa3e4-7f99-41d0-af10-6407caf22f06)

2. Push it onto MONGO DATABASE to store the value

![mongo_database](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/0519fb4f-11f0-4306-bc76-0a339001aa47)

3. Pull the stored values in MONGO DATABASE to get the predicted summarization using SUMMARIZATION SYSTEM

![text_summarization](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/a7b72e62-9d45-4a48-8627-079fefdba168)

4. Deploy it by using FASTAPI

![fast_api](https://github.com/lakiet1609/Text-Recognition-and-Summarization-System/assets/116550803/55b11cf1-0d73-457c-96ee-5bb5eb087129)


