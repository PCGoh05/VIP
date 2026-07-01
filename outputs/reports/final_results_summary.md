# Final Results Summary

This summary was generated from `outputs/reports/model_comparison.csv`.

## Model Comparison

| Model | Accuracy | Macro F1-score | Weighted F1-score | Model size | Avg inference time |
|---|---:|---:|---:|---:|---:|
| simple_cnn | 91.84% | 91.66% | 91.75% | 1.33 MB | 0.0021 s |
| mobilenetv2 | 95.30% | 95.22% | 95.26% | 9.31 MB | 0.0029 s |
| mobilenetv2_finetuned | 96.49% | 96.44% | 96.47% | 20.84 MB | 0.0024 s |
| efficientnetb0 | 96.81% | 96.76% | 96.80% | 16.39 MB | 0.0045 s |

## Main Observation

The best result in this run is `efficientnetb0` with accuracy 96.81% and macro F1-score 96.76%.

MobileNetV2 fine-tuned remains the main required transfer-learning model for the coursework. EfficientNetB0 is reported as an optional additional experiment.

## Honest Limitation

The models perform best on images similar to the selected Kaggle/PlantVillage-style dataset. Real-world images with different lighting, background, camera distance or symptom appearance may be harder and should be treated as preliminary screening only.
