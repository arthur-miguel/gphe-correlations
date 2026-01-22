# GPHE Thermohydraulic Performance Prediction

### Experimental and Machine Learning-Based Predictions of Thermohydraulic Performance in Gasketed Plate Heat Exchangers


This repository contains the official implementation and trained models for predicting the friction factors ($f$) and Nusselt numbers ($Nu$) of Gasketed Plate Heat Exchangers (GPHEs). The models are based on extensive experimental data covering three distinct corrugation geometries: Chevron, Zigzag, and Four-Quadrant.

Pre-trained MLPs exported to ONNX format are provided in the `pretrained` folder, allowing for inference in Python, MATLAB, and other frameworks without requiring the original training environment. If you want to train our models in our data, source code is also provided.

### Features:
* Multi-Geometry Support: Predicts performance for Chevron, Zigzag, and Segmented (Four-Quadrant) plates.

* Assembly-Aware: Explicitly accounts for tightening levels ($A$) and inlet pressure asymmetry ($\Delta P_{in}$), capturing the "breathing effect" often neglected in classical correlations.

* High Accuracy: The provided MLP model outperforms traditional symbolic correlations, even in extrapolation regimes (verified via LOGOCV).

* Cross-Platform: Models are provided in .onnx format for easy deployment.

### Repo Structure

```
├── README.md
├── figures/                        # Plots of model performance and data analysis
├── output/                         # Prediction results and cross-validation logs
├── pretrained/                     # Trained models ready for inference
│   ├── final_model.onnx            # Trained MLP model (Universal format)
│   ├── final_model.pt              # Original PyTorch model state
│   ├── final_model_scalers.json    # Normalization parameters (Mean/Std) for MATLAB
│   └── ... (.joblib scalers for Python)
├── symbolic_reg_f.py               # Symbolic regression script for friction factor
├── symbolic_reg_nu.py              # Symbolic regression script for Nusselt number
├── train_finalModel.py             # Script to train the final model on full dataset
└── train_gpu.py                    # Main training logic with GPU acceleration
```

### Examples

#### Python

Requirements:

```bash
onnxruntime numpy
```

Inference:
```python
import onnxruntime as ort
import json
import numpy as np

# 1. Load Model and Scalers
sess = ort.InferenceSession("pretrained/final_model.onnx")
with open("pretrained/final_model_scalers.json", "r") as f:
    scalers = json.load(f)

# 2. Prepare Input (1 Sample)
# Features: [Re, Delta_P_in, Tightening_A, is_Chevron, is_Zigzag, is_4Quadrant]
X = np.array([[1500.0, 0.5, 1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

# 3. Pre-process (Standard Scaling)
x_mean = np.array(scalers["x_mean"], dtype=np.float32)
x_scale = np.array(scalers["x_scale"], dtype=np.float32)
X_scaled = (X - x_mean) / x_scale

# 4. Run Inference
input_name = sess.get_inputs()[0].name
output_scaled = sess.run(None, {input_name: X_scaled})[0]

# 5. Post-process (Inverse Scaling)
y_mean = np.array(scalers["y_mean"], dtype=np.float32)
y_scale = np.array(scalers["y_scale"], dtype=np.float32)
final_prediction = (output_scaled * y_scale) + y_mean

print(f"Predicted Friction Factors: {final_prediction}")
```


#### MATLAB

Requeriments: 

_Deep Learning Toolbox Converter for ONNX Model Format (Install via Add-On Explorer)_

Inference:

```MATLAB
% 1. Load Model and Scalers
net = importNetworkFromONNX('pretrained/final_model.onnx');

fid = fopen('pretrained/final_model_scalers.json');
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
scalers = jsondecode(str);

% 2. Prepare Input
% Features: [Re, Delta_P_in, Tightening_A, is_Chevron, is_Zigzag, is_4Quadrant]
rawInput = [1500, 0.5, 1.0, 1, 0, 0]; 

% 3. Pre-process (Scaling)
x_mean = scalers.x_mean'; 
x_scale = scalers.x_scale';
inputScaled = (rawInput - x_mean) ./ x_scale;

% 4. Run Inference
% 'BC' = Batch, Channel (standard for tabular ONNX in MATLAB)
inputDL = dlarray(single(inputScaled), 'BC'); 
outputDL = predict(net, inputDL);
outputScaled = extractdata(outputDL);

% 5. Post-process (Inverse Scaling)
y_mean = scalers.y_mean';
y_scale = scalers.y_scale';
finalPrediction = (outputScaled .* y_scale) + y_mean;

disp('Predicted Friction Factors:');
disp(finalPrediction);
```