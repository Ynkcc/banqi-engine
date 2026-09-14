//! 神经网络模型推理模块 (Neural Network Inference)
//!
//! 包含 PyTorch LibTorch / TorchScript 评估器与 ONNX Runtime 推理引擎。
//! NNUE 量化推理见 `crate::nnue`（不依赖 feature）。

#[cfg(any(feature = "torch", feature = "onnx"))]
pub(crate) mod batch;

#[cfg(feature = "torch")]
pub mod torchscript;

#[cfg(feature = "onnx")]
pub mod onnx;
