//! Re-export candle-core CUDA graph helpers.

#![cfg(feature = "cuda")]

pub use candle_core::cuda_graph::*;
