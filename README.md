# NeuroGrade Pro - Brain Tumor Analysis Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.8+-green.svg) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.12.0-orange.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

A professional medical imaging platform for AI-powered brain tumor detection, segmentation, and volumetric analysis. Built with a custom deep learning model and PACS-like interface for radiologists.

## 🧠 Features

### Core AI Capabilities
- **Custom Deep Learning Model**: Self-trained U-Net architecture for brain tumor segmentation
- **Multi-Class Segmentation**: Detects Necrotic Core, Edema, and Enhancing Tumor regions
- **Real-Time Processing**: ~30 second analysis of full MRI volumes
- **High Accuracy**: Dice score 0.85+ on validation data

### Professional Medical Interface
- **Multi-Modality Viewer**: PACS-like 2×2 grid view (T1, T2, FLAIR, T1CE)
- **Interactive Overlays**: Toggle tumor region visibility on/off
- **3D Visualization**: Interactive tumor models with Plotly
- **Dark Mode**: Radiology-standard interface for clinical environments
- **Volumetric Analysis**: Comprehensive tumor volume calculations and statistics

### Clinical Features
- **Professional Reports**: Automated PDF generation with clinical findings
- **Session Management**: Smart data persistence across app usage
- **Medical-Grade UI**: Clinical blue-gray palette matching hospital PACS systems
- **Responsive Design**: Optimized for medical workstations and tablets

## 🎯 Tumor Classification

| Type | Label | Description | Color |
|------|-------|-------------|-------|
| **NCR/NET** | 1 | Necrotic and Non-enhancing tumor core | 🔴 Red |
| **ED** | 2 | Peritumoral edematous/invaded tissue | 🟢 Green |
| **ET** | 4 | GD-enhancing tumor | 🔵 Blue |

## 🚀 Live Demo

**[Launch NeuroGrade Pro →](your-render-url-here)**

*Try with the included sample brain scan data*

## 💻 Local Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (required for AI model inference)
- Modern web browser

### Quick Setup
```bash
# Clone repository
git clone https://github.com/yourusername/neurograde-pro.git
cd neurograde-pro

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Alternative Setup (Virtual Environment)
```bash
# Create virtual environment
python -m venv neurograde_env
source neurograde_env/bin/activate  # On Windows: neurograde_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

## 🏥 Usage Guide

### 1. Upload MRI Data
- Supported formats: NIfTI (.nii, .nii.gz)
- Upload T1, T2, FLAIR, or T1CE sequences
- Use provided sample data to test

### 2. AI Analysis
- Click "Run AI Analysis" 
- Processing time: ~30-60 seconds
- Model automatically segments tumor regions

### 3. Multi-Modality Viewing
- Switch to "Multi Viewer" tab
- View all MRI sequences simultaneously
- Toggle tumor overlays on/off
- Navigate through brain slices

### 4. 3D Visualization
- Interactive 3D tumor reconstruction
- Rotate, zoom, and explore
- Color-coded tumor regions

### 5. Generate Reports
- Professional PDF reports
- Volumetric measurements
- Clinical findings summary
- Ready for medical review

## 🔬 Technical Architecture

### AI/ML Stack
- **Model**: Custom 3D U-Net architecture
- **Framework**: TensorFlow 2.12.0
- **Training**: BraTS dataset (Brain Tumor Segmentation)
- **Preprocessing**: Intensity normalization, skull stripping
- **Postprocessing**: Connected component analysis, morphological operations

### Frontend Stack
- **Interface**: Streamlit (medical-optimized UI)
- **Visualization**: Matplotlib, Plotly 3D
- **Medical Imaging**: NiBabel, scikit-image
- **Styling**: Custom CSS with clinical color palette

### Performance Metrics
- **Processing Speed**: 30-60 seconds per 3D volume
- **Memory Usage**: ~6-8GB during inference
- **Model Size**: 85MB (brain_tumor_unet_final.h5)
- **Dice Coefficient**: 0.85+ average across tumor classes

## 📊 Model Details

### Architecture
```
3D U-Net with:
- Encoder: 5 levels with 3D convolutions
- Decoder: Skip connections + upsampling
- Output: 4-class segmentation (background + 3 tumor types)
- Activation: ReLU (hidden), Softmax (output)
- Loss: Dice + Categorical Crossentropy
```

### Training
- **Dataset**: BraTS 2020 Challenge Dataset
- **Samples**: 369 training cases
- **Augmentation**: Random rotation, flipping, elastic deformation
- **Validation**: 80/20 train/validation split
- **Hardware**: NVIDIA GPU training

## 🎨 UI/UX Design

### Medical-Grade Interface
- **Color Palette**: Clinical blue (#3498DB) and medical grays
- **Typography**: Inter font for clinical readability
- **Layout**: PACS-inspired design patterns
- **Dark Mode**: Eye-strain reduction for radiologists
- **Responsive**: Desktop and tablet optimized

### User Experience
- **Workflow**: Familiar radiology patterns
- **Navigation**: Intuitive medical software UX
- **Feedback**: Real-time processing indicators
- **Accessibility**: High contrast, readable fonts

## 🌍 Built in Malaysia 🇲🇾

Developed to advance global healthcare accessibility and democratize medical AI technology for developing healthcare systems.

## 📁 Project Structure

```
neurograde-pro/
├── app.py                          # Main Streamlit application
├── brain_tumor_unet_final.h5       # Custom trained AI model
├── multi_viewer/
│   └── multi_viewer.py             # Multi-modality viewer module
├── requirements.txt                # Python dependencies
├── Brain Sample Scan.zip           # Sample MRI data
├── README.md                       # This documentation
└── .gitignore                      # Git ignore patterns
```

## 🧪 Sample Data

The included `Brain Sample Scan.zip` contains sample MRI data for testing:
- T1-weighted image
- T2-weighted image  
- FLAIR sequence
- T1CE (contrast-enhanced)

Extract and upload to test the full analysis pipeline.

## 🚀 Deployment

### Render Deployment
1. Push to GitHub repository
2. Connect to Render
3. Deploy as web service
4. Environment: Python 3.8+
5. Build command: `pip install -r requirements.txt`
6. Start command: `streamlit run app.py`

### Docker Deployment
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/neurograde-pro.git
cd neurograde-pro

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install in development mode
pip install -r requirements.txt
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] DICOM format support
- [ ] Multi-language interface
- [ ] Cloud deployment options
- [ ] API endpoints for integration
- [ ] Mobile-responsive improvements
- [ ] Advanced visualization features

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/neurograde-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/neurograde-pro/discussions)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## 🏆 Acknowledgments

- **BraTS Challenge**: For providing the brain tumor dataset
- **Medical AI Community**: For open-source medical imaging tools
- **Streamlit Team**: For the amazing web app framework
- **Malaysia Healthcare**: For inspiring accessible medical technology

## ⚖️ Medical Disclaimer

**IMPORTANT**: This software is for research and educational purposes only. It is NOT intended for clinical diagnosis, treatment planning, or patient care. Always consult qualified medical professionals and follow established clinical protocols for medical decisions.

---

<div align="center">

**🧠 NeuroGrade Pro - Advancing Brain Tumor Analysis with AI** 🚀

*Made with ❤️ in Malaysia 🇲🇾*

[⭐ Star this repo](https://github.com/yourusername/neurograde-pro) | [🐛 Report Bug](https://github.com/yourusername/neurograde-pro/issues) | [💡 Request Feature](https://github.com/yourusername/neurograde-pro/issues)

</div>
