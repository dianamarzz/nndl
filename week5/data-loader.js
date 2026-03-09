/**
 * data-loader.js – parse MNIST csv files (no header, label + 784 ints)
 * provides loadTrainFromFiles, loadTestFromFiles, splitTrainVal,
 * getRandomTestBatch, draw28x28ToCanvas, and safe tensor disposal.
 */

// ---- core CSV parsing (streaming not required, but robust for ~100k rows) ----
async function parseCSVFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                const rows = [];
                for (let line of lines) {
                    const values = line.split(',').map(v => {
                        const num = Number(v.trim());
                        return isNaN(num) ? 0 : num;
                    });
                    if (values.length !== 785) continue; // safety: label + 784
                    rows.push(values);
                }
                resolve(rows);
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file); // read as text – ok for typical MNIST size
    });
}

// Convert parsed rows to tensors: xs [N,28,28,1], ys one‑hot [N,10]
function rowsToTensors(rows) {
    return tf.tidy(() => {
        if (!rows.length) throw new Error('empty csv');
        const labels = [];
        const pixels = [];
        rows.forEach(row => {
            const label = row[0];
            if (label < 0 || label > 9) return; // invalid skip
            labels.push(label);
            pixels.push(row.slice(1)); // 784 numbers
        });

        const labelTensor = tf.tensor1d(labels, 'int32');
        const ys = tf.oneHot(labelTensor, 10);
        labelTensor.dispose();

        // pixels: [N, 784] → float32 → reshape + normalize
        let xs = tf.tensor2d(pixels, [pixels.length, 784], 'float32');
        xs = xs.div(255.0);               // normalize 0..1
        xs = xs.reshape([pixels.length, 28, 28, 1]); // [N,28,28,1]
        return { xs, ys };
    });
}

// Public API -----------------------------------------------------------------
export async function loadTrainFromFiles(file) {
    const rows = await parseCSVFromFile(file);
    return rowsToTensors(rows);
}

export async function loadTestFromFiles(file) {
    const rows = await parseCSVFromFile(file);
    return rowsToTensors(rows);
}

// Split validation set (using slice – no tf then dispose part of xs/ys)
export function splitTrainVal(xs, ys, valRatio = 0.1) {
    const total = xs.shape[0];
    const valSize = Math.floor(total * valRatio);
    const trainSize = total - valSize;

    return tf.tidy(() => {
        // xs and ys are tensors, we need to slice
        const trainXs = xs.slice([0, 0, 0, 0], [trainSize, 28, 28, 1]);
        const trainYs = ys.slice([0, 0], [trainSize, 10]);
        const valXs = xs.slice([trainSize, 0, 0, 0], [valSize, 28, 28, 1]);
        const valYs = ys.slice([trainSize, 0], [valSize, 10]);

        // Return plain tensors (they will be disposed by caller later)
        return { trainXs, trainYs, valXs, valYs };
    });
}

// Get random batch of 5 test samples (tensors). Also returns their label indices
export function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const total = xs.shape[0];
        if (total === 0) return { images: null, labels: null, indices: [] };
        const indices = [];
        for (let i = 0; i < k; i++) {
            indices.push(Math.floor(Math.random() * total));
        }
        // gather using slice (simpler than gatherND)
        const imageList = [];
        const labelList = [];
        for (let idx of indices) {
            const img = xs.slice([idx, 0, 0, 0], [1, 28, 28, 1]); // [1,28,28,1]
            const lbl = ys.slice([idx, 0], [1, 10]); // [1,10]
            imageList.push(img.squeeze([0])); // [28,28,1]
            labelList.push(lbl.squeeze([0])); // [10]
        }
        // combine along new batch dim? but for preview we keep separate.
        return { images: imageList, labels: labelList, indices };
    });
}

// Utility: draw a 28x28 image tensor onto canvas (grayscale, scale factor)
export function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        const width = 28 * scale;
        const height = 28 * scale;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);

        // tensor shape: [28,28,1] (or [28,28])? ensure float
        let imageTensor = tensor;
        if (tensor.shape.length === 3 && tensor.shape[2] === 1) {
            imageTensor = tensor.squeeze([2]); // [28,28]
        }
        const data = imageTensor.mul(255).clipByValue(0, 255).cast('int32').arraySync(); // 2D array

        const imgData = ctx.createImageData(width, height);
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                const val = data[i][j]; // 0-255
                for (let dx = 0; dx < scale; dx++) {
                    for (let dy = 0; dy < scale; dy++) {
                        const px = (i * scale + dx) * 4 * width + (j * scale + dy) * 4;
                        imgData.data[px] = val;
                        imgData.data[px + 1] = val;
                        imgData.data[px + 2] = val;
                        imgData.data[px + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    });
}

// helper to dispose list of tensors
export function disposeTensors(tt) {
    if (!tt) return;
    if (Array.isArray(tt)) tt.forEach(t => t?.dispose());
    else if (tt instanceof tf.Tensor) tt.dispose();
}