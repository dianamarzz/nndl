/**
 * data-loader.js
 * CSV parsing, tensor creation, normalisation, one‑hot, utilities.
 * No network fetch – all from user‑provided File objects.
 * Uses FileReader with chunking for large files (stream‑like via text()).
 * For simplicity we use .text() which is fine for MNIST sizes (<200MB).
 */

// ---------- core: parse CSV text to tensors ----------
async function parseCsvToTensors(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                if (lines.length === 0) throw new Error('empty CSV');

                const rows = [];
                for (let line of lines) {
                    const values = line.split(',').map(Number);
                    if (values.length !== 785) continue; // safety: ignore malformed rows
                    const label = values[0];
                    const pixels = values.slice(1, 785); // 784 values
                    rows.push({ label, pixels });
                }

                if (rows.length === 0) throw new Error('no valid rows');

                // stack pixels and labels
                const numExamples = rows.length;
                const imagesArray = new Float32Array(numExamples * 784);
                const labelsArray = new Int32Array(numExamples);

                rows.forEach((row, idx) => {
                    labelsArray[idx] = row.label;
                    for (let i = 0; i < 784; i++) {
                        // normalize pixel values from 0-255 to 0-1
                        imagesArray[idx * 784 + i] = row.pixels[i] / 255.0;
                    }
                });

                // create tensors: images -> [N,28,28,1]
                const imagesTensor = tf.tensor4d(imagesArray, [numExamples, 28, 28, 1]);
                // one‑hot labels depth 10 (convert to float32 for training)
                const labelsTensor = tf.oneHot(tf.tensor1d(labelsArray, 'int32'), 10).toFloat();

                resolve({ xs: imagesTensor, ys: labelsTensor });
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('file read failed'));
        reader.readAsText(file); // read as text (could use stream but fine for MNIST)
    });
}

// public API
async function loadTrainFromFiles(file) {
    return parseCsvToTensors(file);
}

async function loadTestFromFiles(file) {
    return parseCsvToTensors(file);
}

// simple random split 90/10 with proper tensor indices
function splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
        const num = xs.shape[0];
        const numVal = Math.floor(num * valRatio);
        const numTrain = num - numVal;

        // generate random permutation indices as a tensor
        const indices = tf.util.createShuffledIndices(num);
        // convert to tensor for gather operation
        const indicesTensor = tf.tensor1d(indices, 'int32');
        
        // split indices
        const trainIndices = indicesTensor.slice([0], [numTrain]);
        const valIndices = indicesTensor.slice([numTrain], [numVal]);

        // gather samples using tensor indices
        const trainXs = tf.gather(xs, trainIndices);
        const trainYs = tf.gather(ys, trainIndices);
        const valXs = tf.gather(xs, valIndices);
        const valYs = tf.gather(ys, valIndices);

        // clean up indices tensor
        indicesTensor.dispose();
        
        return { trainXs, trainYs, valXs, valYs };
    });
}

// get k random test samples for preview (tensors)
function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const total = xs.shape[0];
        if (total === 0) return null;
        
        // generate random indices as tensor
        const indices = [];
        for (let i = 0; i < k; i++) {
            indices.push(Math.floor(Math.random() * total));
        }
        const indicesTensor = tf.tensor1d(indices, 'int32');
        
        const batchXs = tf.gather(xs, indicesTensor);
        const batchYs = tf.gather(ys, indicesTensor);
        
        indicesTensor.dispose();
        return { xs: batchXs, ys: batchYs };
    });
}

// draw 28x28 [0..1] tensor to canvas (grayscale)
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        // squeeze to remove batch and channel dimensions if present
        let data = tensor;
        if (tensor.shape.length === 4) {
            data = tensor.squeeze([0, 3]); // remove batch and channel -> [28,28]
        } else if (tensor.shape.length === 3) {
            data = tensor.squeeze([2]); // remove channel if [28,28,1]
        }
        
        const dataArr = data.arraySync(); // 2D array
        const ctx = canvas.getContext('2d');
        const width = 28 * scale;
        const height = 28 * scale;
        canvas.width = width;
        canvas.height = height;

        // create ImageData
        const imgData = ctx.createImageData(width, height);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                // ensure value is in 0-255 range
                let val = dataArr[y][x] * 255;
                val = Math.max(0, Math.min(255, val)); // clamp
                const gray = Math.floor(val);
                
                // fill scaled block (pixelated look)
                for (let dy = 0; dy < scale; dy++) {
                    for (let dx = 0; dx < scale; dx++) {
                        const px = x * scale + dx;
                        const py = y * scale + dy;
                        const idx = (py * width + px) * 4;
                        imgData.data[idx] = gray;
                        imgData.data[idx + 1] = gray;
                        imgData.data[idx + 2] = gray;
                        imgData.data[idx + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    });
}

// export globally (for app.js)
window.loadTrainFromFiles = loadTrainFromFiles;
window.loadTestFromFiles = loadTestFromFiles;
window.splitTrainVal = splitTrainVal;
window.getRandomTestBatch = getRandomTestBatch;
window.draw28x28ToCanvas = draw28x28ToCanvas;