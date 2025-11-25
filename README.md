# SAM_XMem_TensorRT

**基于 SAM、XMem 和 TensorRT 的高性能目标跟踪系统**

## 🚀 项目概览

本项目集成了 **Segment Anything Model (SAM)** 用于零样本（Zero-Shot）目标初始化，以及 **XMem** 用于高精度的视频目标分割。为了实现实时性能，XMem 模型通过 **TensorRT** 进行了加速，从而在消费级 GPU 上实现高效的长期跟踪。

系统提供了一个用户友好的 **Gradio** Web 界面，方便用户交互。用户只需上传视频，单击选择目标，即可实时查看跟踪结果。

## ✨ 主要特性

-   **零样本初始化**：利用 SAM 模型，只需简单点击即可选择要跟踪的对象，无需手动绘制掩码。
-   **长期记忆机制**：XMem 的统一记忆架构能有效处理目标遮挡和重现问题。
-   **TensorRT 加速**：针对 XMem 进行推理优化，确保流畅的跟踪性能。
-   **交互式 UI**：基于 Gradio 构建，提供无缝的浏览器端体验。
-   **显存优化**：针对 8GB 显存 GPU 进行了优化，包含自动显存清理和混合精度推理。

## 🛠️ 安装指南

### 前置要求

-   **操作系统**: Windows (已测试) 或 Linux
-   **GPU**: 支持 CUDA 的 NVIDIA 显卡 (推荐 8GB+ 显存)
-   **Python**: 3.8+
-   **CUDA Toolkit**: 11.x 或 12.x (需与 PyTorch 和 TensorRT 版本匹配)

### 安装步骤

1.  **克隆仓库**
    ```bash
    git clone https://github.com/EscoffierZhou/SAM_XMem_TensorRT.git
    cd SAM_XMem_TensorRT
    ```

2.  **安装依赖**
    确保已安装支持 CUDA 的 PyTorch。然后安装其余依赖包：
    ```bash
    pip install -r requirements.txt
    ```
    *(注意：如果缺少 `requirements.txt`，请手动安装 `gradio`, `opencv-python`, `numpy`, `pillow` 等库)*

3.  **构建 TensorRT 引擎 (C++)**
    XMem 的核心推理通过 C++ 加速。你需要编译该项目：
    ```bash
    cd build
    cmake ..
    make
    ```
    *详细编译说明请参考 `CMakeLists.txt` 和 `编译执行方式.txt`。*

4.  **下载模型**
    请将以下模型权重文件放置在项目根目录：
    -   `XMem.pth`: 预训练的 XMem 权重。
    -   `sam_vit_b_01ec64.pth`: SAM ViT-B 检查点。

## 🏃 使用说明

1.  **启动应用**
    运行主 Python 脚本：
    ```bash
    python app.py
    ```

2.  **访问界面**
    打开浏览器并访问 `http://localhost:7860` (端口可能会自动调整)。

3.  **开始跟踪**
    -   **上传**: 拖拽上传视频文件。
    -   **选择**: 在第一帧图像中点击想要跟踪的对象。
    -   **跟踪**: 点击 "Start Tracking" (开始跟踪) 按钮。
    -   **结果**: 观看生成的带有跟踪遮罩的视频。

## 📂 项目结构

-   `app.py`: Gradio 主程序及 Python 逻辑。
-   `src/`: TensorRT 推理的 C++ 源代码。
-   `include/`: 头文件。
-   `models/`: 模型权重目录 (请确保模型按配置放置)。
-   `XMem/`: XMem 子模块/代码库。

## 🤝 贡献

欢迎提交 Pull Request 来改进本项目！

## 📄 许可证

[MIT License](LICENSE)
