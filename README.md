The environment used can be replicated as:
```conda env create -f beampy_nodep.yml```

Data Generation can be done using BeamNG, by running  
```./GenerateData.sh```

Training ZLIK:
```python3 TrainingPipeline/zlik_train.py```


Based on the project structure and training scripts found in your files, here is a draft for your new project's `README.md` following the flow of your previous project:

# ZLIK: "Zero-Shot Language-Informed Kinodynamics"

### Language-Informed Kinodynamics Model for Structurally Damaged Robots

📄 **Checkout our paper:** [Link to Paper (arXiv:XXXX.XXXXX)]

🎥 **Checkout our Video:** [Video Link]

---

*ZLIK enables adaptive kinodynamic modeling for robots with structural damage by leveraging natural language descriptions. This allows for zero-shot adaptation to various damage scenarios, ensuring stable motion planning even under unforeseen structural failures.*

---

## 🧠 Overview

Traditional kinodynamic models often fail when a robot undergoes structural changes or damage. **ZLIK (Zero-Shot Language-Informed Kinodynamics)** addresses this by incorporating high-level natural language descriptions of the robot's state into the dynamics model.

Key features include:

* 
**Zero-Shot Adaptation**: Generalizes to novel damage scenarios without retraining on specific failure cases.


* 
**Language Integration**: Uses a language-informed approach to bias the dynamics model based on text embeddings (e.g., "front right tyre punctured").


* 
**Transformer-Based Architecture**: Employs an encoder-decoder transformer to predict future robot poses based on historical states, future actions, and damage context.



---

## 📦 Installation

The environment can be replicated using Conda:

```bash
conda env create -f beampy_nodep.yml
conda activate ros2_beampy

```

---

## 🛠 Data Generation

Data is generated using the BeamNG simulator, covering various damage scenarios like tyre punctures, axle breaks, and suspension failures.

Run the data generation pipeline:

```bash
./GenerateData.sh

```

This script handles:

1. Running random walks with/without damage.


2. Encoding damage embeddings via `embedding-gemma`.


3. Extracting trajectories for training.



---

## 🚀 Training Pipeline

ZLIK involves a multi-stage training process including pre-training and specialized dynamics training.

### 1. Pre-training Damage Embeddings

```bash
python3 TrainingPipeline/pre_train.py --conf config/pre_train_config.yaml

```

### 2. Training Dynamics Models

You can train specific dynamics models depending on the robot state:

**Clean Dynamics:**

```bash
python3 TrainingPipeline/clean_dynamics_train.py --conf config/damaged_model_config.yaml

```

**ZLIK (Language-Informed):**

```bash
python3 TrainingPipeline/zlik_train.py --conf TrainingPipeline/conf/damaged_model_config.yaml --encoder_conf config/pre_train_config.yaml

```

---

## 📖 Citation

If you find this work useful, please cite our manuscript:

```bibtex
@article{pokhrel2026zlik,
  title={ZLIK: Zero-Shot Language-Informed Kinodynamics for Structurally Damaged Robots},
  author={Pokhrel, Anuj and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}

```
