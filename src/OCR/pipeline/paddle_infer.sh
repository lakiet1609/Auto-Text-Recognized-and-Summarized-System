# python3 components/predict_det.py --det_algorithm="DB" --det_model_dir="models/en_PP-OCRv3_det_infer" --image_dir="common/doc/imgs/ger_2.jpg" --use_gpu=True

python3 components/predict_system.py --det_model_dir="models/en_PP-OCRv3_det_infer" --rec_model_dir="models/en_PP-OCRv3_rec_infer" --use_angle_cls=false --rec_char_dict_path='common/ppocr/utils/en_dict.txt'