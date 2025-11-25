# SAM_XMem_TensorRT

**High-Performance Object Tracking with SAM, XMem, and TensorRT**

![Preview](assets/preview.jpg)

## 🚀 Overview

This project integrates **Segment Anything Model (SAM)** for zero-shot object initialization and **XMem** for high-accuracy video object segmentation. To achieve real-time performance, the XMem model is accelerated using **TensorRT**, enabling efficient long-term tracking on consumer-grade GPUs.

The system provides a user-friendly **Gradio** web interface for easy interaction, allowing users to upload videos, select objects with a single click, and visualize tracking results instantly.

## ✨ Features

-   **Zero-Shot Initialization**: Use SAM to select objects to track with a simple click. No manual mask drawing required.
-   **Long-Term Memory**: XMem's unified memory architecture handles occlusion and re-appearance effectively.
-   **TensorRT Acceleration**: Optimized inference for XMem ensures smooth tracking performance.
-   **Interactive UI**: Built with Gradio for a seamless browser-based experience.
-   **Memory Management**: Optimized for 8GB VRAM GPUs with automatic memory cleanup and mixed-precision inference.

## 🛠️ Installation

### Prerequisites

-   **OS**: Windows (tested) or Linux
-   **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
-   **Python**: 3.8+
-   **CUDA Toolkit**: 11.x or 12.x (matching your PyTorch and TensorRT versions)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/EscoffierZhou/SAM_XMem_TensorRT.git
    cd SAM_XMem_TensorRT
    ```

2.  **Install Dependencies**
    Ensure you have PyTorch installed with CUDA support. Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to install `gradio`, `opencv-python`, `numpy`, `pillow` if `requirements.txt` is not present)*

3.  **Build TensorRT Engine (C++)**
    The core XMem inference is accelerated via C++. You need to build the project:
    ```bash
    cd build
    cmake ..
    make
    ```
    *Refer to `CMakeLists.txt` and `编译执行方式.txt` for detailed build instructions.*

4.  **Download Models**
    Place the following model weights in the project root:
    -   `XMem.pth`: Pre-trained XMem weights.
    -   `sam_vit_b_01ec64.pth`: SAM ViT-B checkpoint.

## 🏃 Usage

1.  **Start the Application**
    Run the main Python script:
    ```bash
    python app.py
    ```

2.  **Access the UI**
    Open your browser and navigate to `http://localhost:7860`.

3.  **Track Objects**
    -   **Upload**: Drag and drop a video file.
    -   **Select**: Click on the object you want to track in the first frame.
    -   **Track**: Click the "Start Tracking" button.
    -   **Result**: Watch the generated video with the tracking overlay.

## 📂 Project Structure

-   `app.py`: Main Gradio application and Python logic.
-   `src/`: C++ source code for TensorRT inference.
-   `include/`: Header files.
-   `models/`: Directory for model weights (ensure models are placed here or in root as configured).
-   `XMem/`: XMem submodule/codebase.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT License](LICENSE)
