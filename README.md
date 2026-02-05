# ML Fine-tuning CI/CD Pipeline

This repository contains a complete CI/CD pipeline for machine learning fine-tuning projects using Python.

## 🚀 Pipeline Overview

The pipeline includes 6 automated jobs:

1. **Code Quality** - Linting and formatting checks (Black, isort, Flake8)
2. **Unit Tests** - Run tests with coverage reporting
3. **Model Validation** - Quick training run on sample data to verify pipeline
4. **Full Training** - Complete fine-tuning (only on main branch)
5. **Model Deployment** - Upload to model registry and deploy
6. **Performance Monitoring** - Check model performance against baseline

## 📋 Prerequisites

- Python 3.10+
- GitHub account (for GitHub Actions)
- GPU recommended for training (can use GitHub-hosted runners or self-hosted)

## 🛠️ Setup

### 1. Clone and Install

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
```

### 2. Configure Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `HF_TOKEN` - Hugging Face token (for model registry)
- `MLFLOW_URI` - MLflow tracking server (optional)
- `AWS_ACCESS_KEY_ID` - For AWS deployment (optional)
- `AWS_SECRET_ACCESS_KEY` - For AWS deployment (optional)

### 3. Project Structure

```
your-project/
├── .github/
│   └── workflows/
│       └── ml-finetune-pipeline.yml  # CI/CD pipeline
├── configs/
│   └── training_config.yaml          # Training configuration
├── scripts/
│   ├── download_data.py              # Data download script
│   ├── validate_model.py             # Model validation
│   ├── upload_model.py               # Upload to registry
│   └── check_performance.py          # Performance monitoring
├── tests/
│   ├── unit/                         # Unit tests
│   └── integration/                  # Integration tests
├── src/                              # Your source code
├── train.py                          # Main training script
├── evaluate.py                       # Evaluation script
└── requirements.txt                  # Python dependencies
```

## 🎯 Usage

### Local Training

```bash
# Quick validation run
python train.py --epochs 1 --batch-size 8 --validation-only

# Full training
python train.py --config configs/training_config.yaml
```

### Trigger CI/CD Pipeline

The pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual trigger (workflow_dispatch)

To manually trigger:
1. Go to Actions tab in GitHub
2. Select "ML Fine-tuning CI/CD Pipeline"
3. Click "Run workflow"

## 📊 Customization

### Modify Training Script

Edit `train.py` to customize:
- Model architecture
- Dataset loading
- Training hyperparameters
- Evaluation metrics

### Adjust Pipeline

Edit `.github/workflows/ml-finetune-pipeline.yml` to:
- Change Python version
- Add/remove jobs
- Modify timeout limits
- Add deployment targets

### Configure Training

Create `configs/training_config.yaml`:

```yaml
model_name: "bert-base-uncased"
epochs: 3
batch_size: 16
learning_rate: 2e-5
output_dir: "outputs/model"
```

## 🔍 Monitoring

- **Code coverage**: Uploaded to Codecov
- **Training logs**: Available in GitHub Actions
- **Model artifacts**: Stored for 30 days
- **Performance metrics**: Saved in `outputs/model/metrics.json`

## 🎓 Learning Resources

### Fine-tuning Concepts

- **Transfer Learning**: Using pre-trained models as starting point
- **Fine-tuning**: Adjusting model weights for your specific task
- **Hyperparameter Tuning**: Finding optimal learning rate, batch size, etc.
- **Evaluation**: Measuring model performance on validation data

### Recommended Reading

- [Hugging Face Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)
- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ML CI/CD Best Practices](https://ml-ops.org/content/mlops-principles)

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests locally: `pytest tests/`
4. Submit a pull request

The CI/CD pipeline will automatically validate your changes!

## 📝 Next Steps

1. ✅ Set up GitHub repository
2. ✅ Configure secrets
3. ⬜ Customize `train.py` for your dataset
4. ⬜ Add unit tests in `tests/`
5. ⬜ Create training config in `configs/`
6. ⬜ Push to GitHub and watch the magic happen! 🎉

## 💡 Tips

- Start with validation-only runs to test your pipeline quickly
- Use small datasets initially to iterate faster
- Monitor GPU usage and adjust batch sizes accordingly
- Version your datasets and models for reproducibility
- Set up experiment tracking with Weights & Biases or MLflow

Happy fine-tuning! 🚀
