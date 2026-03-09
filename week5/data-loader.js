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
                    if (values.length !== 785) continue; // safety: ignore malformed
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
                        imagesArray[idx * 784 + i] = row.pixels[i] / 255.0; // normalize
                    }
                });

                // create tensors: images -> [N,28,28,1]
                const imagesTensor = tf.tensor4d(imagesArray, [numExamples, 28, 28, 1]);
                // one‑hot labels depth 10
                const labelsTensor = tf.oneHot(tf.tensor1d(labelsArray, 'int32'), 10);

                resolve({ xs: imagesTensor, ys: labelsTensor });
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('file read failed'));
        reader.readAsText(file); // read as text (could use stream but ok)
    });
}

// public API
async function loadTrainFromFiles(file) {
    return parseCsvToTensors(file);
}

async function loadTestFromFiles(file) {
    return parseCsvToTensors(file);
}

// simple random split 90/10 (no shuffle – user can shuffle before)
function splitTrainVal(xs, ys, valRatio = 0.1) {
    const num = xs.shape[0];
    const numVal = Math.floor(num * valRatio);
    const numTrain = num - numVal;

    // generate random permutation and pick last numVal as validation
    const indices = tf.util.createShuffledIndices(num);
    const trainIndices = indices.slice(0, numTrain);
    const valIndices = indices.slice(numTrain, num);

    const trainXs = tf.gather(xs, trainIndices);
    const trainYs = tf.gather(ys, trainIndices);
    const valXs = tf.gather(xs, valIndices);
    const valYs = tf.gather(ys, valIndices);

    return { trainXs, trainYs, valXs, valYs };
}

// get k random test samples for preview (tensors)
function getRandomTestBatch(xs, ys, k = 5) {
    const total = xs.shape[0];
    if (total === 0) return null;
    const indices = [];
    for (let i = 0; i < k; i++) {
        indices.push(Math.floor(Math.random() * total));
    }
    const batchXs = tf.gather(xs, indices);
    const batchYs = tf.gather(ys, indices);
    return { xs: batchXs, ys: batchYs };
}

// draw 28x28 [0..1] tensor to canvas (grayscale)
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        // squeeze to 2d if needed
        const data = tensor.squeeze(); // shape [28,28] or [1,28,28,1]? handle
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
                const val = dataArr[y][x] * 255; // 0..1 -> 0..255
                const gray = Math.floor(val);
                // fill scaled block
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