// data-loader.js – parse MNIST CSV files into tensors (no network)

/**
 * Parse a CSV file (File object) into an array of rows.
 * Robust: handles large files with streaming via FileReader + chunking (simple readAsText fallback works for MNIST ~100MB)
 */
async function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const text = e.target.result;
            const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
            resolve(lines);
        };
        reader.onerror = () => reject(new Error('File read failed'));
        reader.readAsText(file); // safe for typical MNIST csv (under 200MB)
    });
}

/**
 * Convert parsed CSV lines into tensors: xs [N,28,28,1], ys one‑hot [N,10]
 * Normalize pixels to [0,1] and ensure proper data types
 */
function linesToTensors(lines) {
    return tf.tidy(() => {
        const numSamples = lines.length;
        const labelsArray = [];
        const imagesArray = [];

        for (let line of lines) {
            const values = line.split(',').map(Number);
            if (values.length !== 785) continue; // malformed skip
            
            const label = values[0];
            // Normalize pixels to [0,1] and convert to Float32
            const pixels = values.slice(1).map(v => v / 255.0);
            
            labelsArray.push(label);
            imagesArray.push(pixels);
        }

        console.log(`Parsed ${imagesArray.length} samples, ${imagesArray[0]?.length} features`);

        // Create tensors with proper data types
        const imagesTensor = tf.tensor2d(imagesArray, [imagesArray.length, 784], 'float32');
        const xs = imagesTensor.reshape([-1, 28, 28, 1]);
        
        // Create labels as one-hot with proper int32 type
        const labelsTensor = tf.tensor1d(labelsArray, 'int32');
        const ys = tf.oneHot(labelsTensor, 10);
        
        return { xs, ys };
    });
}

/** loadTrainFromFiles: accept File object → {xs, ys} */
export async function loadTrainFromFiles(file) {
    if (!file) throw new Error('No train file provided');
    console.log('Loading train file:', file.name);
    const lines = await parseCSVFile(file);
    console.log(`Train file has ${lines.length} lines`);
    return linesToTensors(lines);
}

/** loadTestFromFiles: same as above */
export async function loadTestFromFiles(file) {
    if (!file) throw new Error('No test file provided');
    console.log('Loading test file:', file.name);
    const lines = await parseCSVFile(file);
    console.log(`Test file has ${lines.length} lines`);
    return linesToTensors(lines);
}

/**
 * splitTrainVal: take xs, ys and split by valRatio (default 0.1)
 * returns {trainXs, trainYs, valXs, valYs}
 */
export function splitTrainVal(xs, ys, valRatio = 0.1) {
    return tf.tidy(() => {
        const numSamples = xs.shape[0];
        const valSize = Math.floor(numSamples * valRatio);
        const trainSize = numSamples - valSize;

        // Create random permutation of indices
        const indices = tf.util.createShuffledIndices(numSamples);
        
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
 * getRandomTestBatch: returns {xs, ys, indices} tensors for k random samples (preserve for drawing)
 */
export function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const total = xs.shape[0];
        const randIndices = [];
        for (let i = 0; i < k; i++) {
            randIndices.push(Math.floor(Math.random() * total));
        }
        const batchXs = tf.gather(xs, randIndices);
        const batchYs = tf.gather(ys, randIndices);
        return { xs: batchXs, ys: batchYs, indices: randIndices };
    });
}

/**
 * draw28x28ToCanvas: tensor shape [1,28,28,1] or [28,28,1] → scale factor (default 4)
 */
export function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        let squeezed = tensor.squeeze();
        if (squeezed.rank === 2) {
            // already 28x28
        } else if (squeezed.rank === 3 && squeezed.shape[2] === 1) {
            squeezed = squeezed.squeeze([2]); // to [28,28]
        } else {
            throw new Error('Tensor shape not compatible');
        }
        
        const width = 28 * scale;
        const height = 28 * scale;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;

        const pixelValues = squeezed.arraySync(); // Get as array for better performance

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const srcX = Math.floor(x / scale);
                const srcY = Math.floor(y / scale);
                const val = pixelValues[srcY][srcX];
                const gray = Math.round(val * 255);
                const idx = (y * width + x) * 4;
                data[idx] = gray;
                data[idx + 1] = gray;
                data[idx + 2] = gray;
                data[idx + 3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);
    });
}