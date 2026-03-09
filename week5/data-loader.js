// data-loader.js – file parsing, tensor creation, utilities
// all functions return tensors (no network), dispose intermediate results.

/**
 * Parse CSV file (as text) into {xs, ys} tensors.
 * @param {File} file - .csv file (label + 784 pixels)
 * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>}
 */
async function loadTrainFromFiles(file) {
    return parseCsvFile(file, true);   // true = normalize & reshape for training
}

/**
 * Same as above but for test – same parsing logic.
 */
async function loadTestFromFiles(file) {
    return parseCsvFile(file, true);
}

/**
 * Core parser: read file → rows → build tensors, normalize, reshape, one-hot.
 * @param {File} file - input CSV file
 * @param {boolean} normalizeAndReshape - always true (we need image format)
 * @returns {Promise<{xs: tf.Tensor, ys: tf.Tensor}>}
 */
async function parseCsvFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                const numRows = lines.length;
                if (numRows === 0) throw new Error('empty file');

                // Prepare arrays: labels (int) and pixels (float)
                const labels = [];
                const pixels = [];

                for (let line of lines) {
                    const values = line.split(',').map(v => parseFloat(v.trim()));
                    if (values.length !== 785) {
                        console.warn('skipping row, expected 785 numbers, got', values.length);
                        continue;
                    }
                    const label = values[0];
                    if (label < 0 || label > 9) throw new Error(`invalid label: ${label}`);
                    labels.push(label);
                    pixels.push(values.slice(1)); // 784 values
                }

                // one‑hot encode labels (depth 10)
                const ys = tf.tidy(() => {
                    const labelsTensor = tf.tensor1d(labels, 'int32');
                    return tf.oneHot(labelsTensor, 10);
                });

                // pixels: normalize 0-255 → 0-1, reshape to [N,28,28,1]
                const xs = tf.tidy(() => {
                    const pixelsTensor = tf.tensor2d(pixels, [labels.length, 784]); // [N, 784]
                    const normalized = pixelsTensor.div(255.0);          // float [0,1]
                    return normalized.reshape([labels.length, 28, 28, 1]);
                });

                resolve({ xs, ys });
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('file read failed'));
        reader.readAsText(file, 'UTF-8');
    });
}

/**
 * Split training data into train/validation sets.
 * @param {tf.Tensor} xs
 * @param {tf.Tensor} ys
 * @param {number} valRatio
 * @returns {{trainXs: tf.Tensor, trainYs: tf.Tensor, valXs: tf.Tensor, valYs: tf.Tensor}}
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
    const total = xs.shape[0];
    const valSize = Math.floor(total * valRatio);
    const trainSize = total - valSize;

    return tf.tidy(() => {
        // shuffle indices to get random split
        const indices = tf.util.createShuffledIndices(total);
        const trainIndices = indices.slice(0, trainSize);
        const valIndices = indices.slice(trainSize);

        const trainXs = tf.gather(xs, trainIndices);
        const trainYs = tf.gather(ys, trainIndices);
        const valXs = tf.gather(xs, valIndices);
        const valYs = tf.gather(ys, valIndices);

        return { trainXs, trainYs, valXs, valYs };
    });
}

/**
 * Get a small random batch of test images and labels.
 * @param {tf.Tensor} xs test images [N,28,28,1]
 * @param {tf.Tensor} ys one-hot labels
 * @param {number} k how many samples
 * @returns {{batchXs: tf.Tensor, batchYs: tf.Tensor}}
 */
function getRandomTestBatch(xs, ys, k = 5) {
    const total = xs.shape[0];
    if (total === 0) throw new Error('empty test set');
    const sampleSize = Math.min(k, total);
    // pick random indices (no replacement)
    const indices = [];
    while (indices.length < sampleSize) {
        const r = Math.floor(Math.random() * total);
        if (!indices.includes(r)) indices.push(r);
    }
    return tf.tidy(() => {
        const batchXs = tf.gather(xs, indices);
        const batchYs = tf.gather(ys, indices);
        return { batchXs, batchYs };
    });
}

/**
 * Render a 28x28 grayscale tensor onto a canvas (scaled).
 * @param {tf.Tensor} imgTensor shape [28,28] or [28,28,1] (float 0-1)
 * @param {HTMLCanvasElement} canvas
 * @param {number} scale factor (default 4 → 112x112)
 */
function draw28x28ToCanvas(imgTensor, canvas, scale = 4) {
    return tf.tidy(() => {
        const tensor = imgTensor.squeeze(); // [28,28]
        const width = 28 * scale;
        canvas.width = width;
        canvas.height = width;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, width);

        const data = tensor.dataSync(); // flat float32 [0..1]
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const val = Math.floor(data[y * 28 + x] * 255); // grayscale
                for (let dy = 0; dy < scale; dy++) {
                    for (let dx = 0; dx < scale; dx++) {
                        const px = (y * scale + dy) * width + (x * scale + dx);
                        const idx = px * 4;
                        imageData.data[idx] = val;     // R
                        imageData.data[idx + 1] = val; // G
                        imageData.data[idx + 2] = val; // B
                        imageData.data[idx + 3] = 255; // A
                    }
                }
            }
        }
        ctx.putImageData(imageData, 0, 0);
    });
}

// export global functions (used by app.js)
window.dataLoader = {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    getRandomTestBatch,
    draw28x28ToCanvas
};