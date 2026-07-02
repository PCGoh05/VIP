# Final Results Summary

## Dataset

- Selected classes: 8 tomato, potato and corn disease classes
- Training images: 14,808
- Validation images: 1,851
- Final test images: 1,851
- Unreadable images detected: 0

## Model Comparison on Final Test Set

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Model Size | Avg Inference Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple CNN | 91.84% | 92.62% | 91.54% | 91.66% | 91.75% | 1.33 MB | 0.0049 s/image |
| MobileNetV2 | 95.25% | 95.21% | 95.22% | 95.17% | 95.21% | 9.31 MB | 0.0222 s/image |
| MobileNetV2 fine-tuned | 96.43% | 96.56% | 96.43% | 96.38% | 96.42% | 20.84 MB | 0.0222 s/image |

## Main Finding

The fine-tuned MobileNetV2 achieved the best overall classification performance. The Simple CNN is smaller and slightly faster, but the fine-tuned MobileNetV2 is the better main model for the final Streamlit prototype because it has the highest accuracy and strongest macro F1-score.

## Grad-CAM

Grad-CAM examples were generated for 3 correct predictions and 3 incorrect predictions where available. The incorrect examples mainly show confusion between corn gray leaf spot and northern leaf blight, which is useful for the error analysis section.

## Honest Limitation

The dataset images are mostly controlled leaf images, so the prototype is suitable for educational and preliminary screening purposes only. It should not be presented as a replacement for expert agricultural diagnosis.
