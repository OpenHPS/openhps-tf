export * from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';

// Register a hardware-accelerated backend for Node: prefer the CUDA build,
// fall back to CPU-only on machines without a CUDA runtime.
//
// Note: the WebGPU/WebGL (Dawn/headless-gl) backends were investigated as an
// alternative for the Blackwell (RTX 5060 Ti) cards, which tfjs-node-gpu's
// libtensorflow 2.9.1 cannot target. On this box they do not reach the GPU
// (NVIDIA Vulkan is GLX-fused and cannot initialise headless; headless-gl EGL
// fails), so they are not wired up here.
export function ensureBackend(): Promise<void> {
    return tf.ready();
}

try {
    require('@tensorflow/tfjs-node-gpu');
} catch (err) {
    require('@tensorflow/tfjs-node');
}

