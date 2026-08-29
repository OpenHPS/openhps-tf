// Verifies that tfjs-node-gpu actually runs a kernel on the GPU.
// Used by scripts/remote_setup.sh; exits non-zero on failure.
// A device merely being *listed* is not proof: an unsupported architecture
// shows up and then fails at kernel launch ("no kernel image is available").
const tf = require('@tensorflow/tfjs-node-gpu');
const a = tf.tensor2d([[1, 2], [3, 4]]);
const out = a.matMul(a).dataSync();
console.log('backend:', tf.getBackend());
console.log('matmul result:', Array.from(out));
if (tf.getBackend() !== 'tensorflow') {
    console.error('!! not running on the tensorflow backend');
    process.exit(1);
}
if (Math.abs(out[0] - 7) > 1e-6) {
    console.error('!! matmul produced a wrong result (likely CPU fallback or NaN)');
    process.exit(1);
}
console.log('GPU verification OK');
