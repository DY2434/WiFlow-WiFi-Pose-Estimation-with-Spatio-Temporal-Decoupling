# WiFlow: Continuous Human Pose Estimation via WiFi Signals

Existing WiFi-based human pose estimation methods often focus on discrete pose sample evaluation. This work presents WiFlow, a novel framework for continuous human pose estimation using WiFi signals. Unlike vision-based approaches such as two-dimensional deep residual networks that treat Channel State Information (CSI) as images, WiFlow employs an encoder-decoder architecture. The encoder captures spatio-temporal features of CSI using temporal and asymmetric convolutions, preserving the original sequential structure of signals. It then refines keypoint features of human bodies to be tracked and capture their structural dependencies via axial attention. The decoder subsequently maps the encoded high-dimensional features into keypoint coordinates.

* [Architecture](#architecture)
* [Dataset](#dataset)
* [Usage](#usage)
* [Visualization](#visualization)
* [Result](#result)

---

## Architecture

![Architecture](architecture.jpg)

---

## Dataset

**Download Link**: [https://kaggle.com/datasets/5dc84daab11fab92e4a98f4ecf9fbf5ab9a32ca0c101074b602e98b4b33e2222]

**Dataset Statistics**:
- Total Samples: 360,000 synchronized CSI-pose samples
- Subjects: 5
- Activities: walking, raising hands, squatting, hands up, kicking, waving, turning, and jumping
- Keypoints: 15 body joints
- CSI Dimensions: 540 × 20

---

## Usage

### Installation

Clone the repository:
```bash
git clone https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling.git
cd yourrepo
```

### Setup Environment

Create virtual environment and install dependencies:
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

Available arguments:
- `--gpu`: GPU configuration (default: '0')
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--output_dir`: Output directory for results (default: 'output')
- `--use_augmentation`: Enable data augmentation

### Training

Train the model with default parameters:
```bash
python run.py
```

### Testing

Trained models are automatically evaluated on the test set after training. Results are saved in the output directory including:
- Best model checkpoint: `best_pose_model.pth`
- Training history: `training_history.csv`
- Test predictions: `test_predictions.csv`
- Visualization videos in `videos/` folder

---

## Visualization

### Comparison with Ground Truth

<table>
    <tr>
        <td align="center"><img src="ture.gif" width="400"></td>
    </tr>
    <tr>
        <td align="center"><img src="visualization.gif" width="400"></td>
    </tr>
</table>

### Training Curves

![Training History](training_history.png)

---

### Visual comparison of OpenPose (top) and WiFlow (bottom) results

![Visual comparison](fig.png)

## Result

### Comparison with Baseline Methods

| Metric                    | **WiFlow** | WiSPPN | WPformer |
|---------------------------|--------|-------|----------|
| **PCK@0.2**               | **97.00**% | 85.87% | 70.02%   |
| **PCK@0.3**               | **98.55**% | 92.23% | 82.98%   |
| **PCK@0.4**               | **99.15**% | 95.52% | 89.33%   |
| **PCK@0.5**               | **99.48**% | 97.48% | 93.22%   |
| **MPE (Mean Pose Error)** | **0.008**  | 0.016 | 0.028    |
| **Param(M)**              | **4.82**   | 121.50 | 10.01    |
| **FLOPs(B)**              | **0.47**   | 338.45 | 35.00    |
| **Time(h)**               | **9.20**   | 68.10 | 35.47    |

### ABLATION STUDY RESULTS

| Method                                                              | PCK@0.1 | PCK@0.2 |
|---------------------------------------------------------------------|---------|---------|
| WiFlow                                                              | 90.64%  | 97.00%  |
| Replace TCN with regular 1D convolution                             | 84.20%  | 96.44%  |
| Replace TCN and Asymmetric Convolution with 2D residual convolution | 83.55%  | 95.69%  |
| Remove Dual Axial Attention | 86.59%  | 96.43%  |

## License

[Apache License 2.0]

---