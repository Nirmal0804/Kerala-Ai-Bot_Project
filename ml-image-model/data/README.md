# Kerala Crop Disease Detection System

TensorFlow + Keras project for detecting crop diseases from leaf images.

## Structure

```text
ml-image-model/
	data/
		README.md
	models/
	src/
		preprocessing/
			data_loader.py
		training/
			train_model.py
		inference/
			predict.py
	requirements.txt
	.gitignore
	test_pipeline.py
```

## Dataset for Kerala Crop Disease Detection

Classes:

banana_sigatoka
banana_leaf_spot
banana_healthy
okra_yellow_vein_mosaic
okra_powdery_mildew
okra_healthy
chilli_leaf_curl
chilli_healthy

Images should be organized in folders where each folder represents one class.

## Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train model.

```bash
python -m src.training.train_model --data-dir data
```

3. Predict one image.

```python
from src.inference.predict import predict_disease

result = predict_disease("path/to/leaf.jpg")
print(result)
```
