/**
 * app.js – wires UI, handles model lifecycle, training / eval / preview.
 * Uses global tf, tfvis, and imported functions from data-loader.js.
 */
import { loadTrainFromFiles, loadTestFromFiles, splitTrainVal, getRandomTestBatch,
         draw28x28ToCanvas, disposeTensors } from './data-loader.js';

// --- global state ---
let trainTensors = null;     // { xs, ys } (full train)
let testTensors = null;      // { xs, ys } (full test)
let currentModel = null;
let trainingStatus = 'idle';

// DOM elements
const trainInput = document.getElementById('trainFileInput');
const testInput = document.getElementById('testFileInput');
const loadDataBtn = document.getElementById('loadDataBtn');
const trainBtn = document.getElementById('trainBtn');
const evaluateBtn = document.getElementById('evaluateBtn');
const testFiveBtn = document.getElementById('testFiveBtn');
const saveModelBtn = document.getElementById('saveModelBtn');
const resetBtn = document.getElementById('resetBtn');
const toggleVisorBtn = document.getElementById('toggleVisorBtn');
const modelJsonFile = document.getElementById('modelJsonFile');
const modelBinFile = document.getElementById('modelBinFile');
const loadModelBtn = document.getElementById('loadModelBtn');
const dataStatusDiv = document.getElementById('dataStatus');
const trainingLogsDiv = document.getElementById('trainingLogs');
const metricsDisplay = document.getElementById('metricsDisplay');
const modelSummaryPre = document.getElementById('modelSummary');
const previewContainer = document.getElementById('previewContainer');

// --- helpers ---
function updateDataStatus() {
    const trainCount = trainTensors?.xs?.shape[0] ?? 0;
    const testCount = testTensors?.xs?.shape[0] ?? 0;
    dataStatusDiv.innerHTML = `✅ train: ${trainCount} samples | test: ${testCount} samples`;
}

function log(message, isError = false) {
    const p = document.createElement('div');
    p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    p.style.color = isError ? '#b91c1c' : '#0f172a';
    trainingLogsDiv.appendChild(p);
    trainingLogsDiv.scrollTop = trainingLogsDiv.scrollHeight;
}

function clearLogs() { trainingLogsDiv.innerHTML = ''; }

function resetState() {
    // dispose tensors and model
    if (trainTensors) { trainTensors.xs?.dispose(); trainTensors.ys?.dispose(); }
    if (testTensors) { testTensors.xs?.dispose(); testTensors.ys?.dispose(); }
    if (currentModel) { currentModel.dispose(); currentModel = null; }
    trainTensors = testTensors = null;
    updateDataStatus();
    modelSummaryPre.innerText = '(no model)';
    previewContainer.innerHTML = '';
    metricsDisplay.innerText = '';
    log('🧹 reset complete');
    tfvis.visor().close(); // optional close
}

function buildModel() {
    tf.tidy(() => { // just for shape checks, not needed for construction
        const model = tf.sequential();
        model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same', inputShape: [28,28,1] }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
        return model;
    });
}

function rebuildModel() {
    if (currentModel) currentModel.dispose();
    currentModel = buildModel();
    // show summary
    currentModel.summary(null, null, (msg) => modelSummaryPre.innerText += msg + '\n');
    modelSummaryPre.innerText = 'Model built:\n' + modelSummaryPre.innerText;
    log('new CNN model created');
}

// --- Load data ---
loadDataBtn.addEventListener('click', async () => {
    const trainFile = trainInput.files[0];
    const testFile = testInput.files[0];
    if (!trainFile || !testFile) {
        alert('Please select both train.csv and test.csv');
        return;
    }
    try {
        log('📂 parsing train file...');
        const train = await loadTrainFromFiles(trainFile);
        log('📂 parsing test file...');
        const test = await loadTrainFromFiles(testFile); // reuse same loader (test file)
        // replace old tensors
        if (trainTensors) { trainTensors.xs?.dispose(); trainTensors.ys?.dispose(); }
        if (testTensors) { testTensors.xs?.dispose(); testTensors.ys?.dispose(); }
        trainTensors = train;
        testTensors = test;
        updateDataStatus();
        log(`train: ${train.xs.shape[0]} | test: ${test.xs.shape[0]}`);
        rebuildModel(); // create fresh model after data load
    } catch (e) {
        console.error(e);
        log('❌ load failed: ' + e.message, true);
    }
});

// --- Train ---
trainBtn.addEventListener('click', async () => {
    if (!trainTensors || !currentModel) { alert('load data first'); return; }
    if (trainingStatus === 'training') return;

    const { xs, ys } = trainTensors;
    const { trainXs, trainYs, valXs, valYs } = splitTrainVal(xs, ys, 0.1);
    log(`training on ${trainXs.shape[0]} samples, val ${valXs.shape[0]}`);

    try {
        trainingStatus = 'training';
        const container = { name: 'MNIST Training', tab: 'Model' };
        const callbacks = tfvis.show.fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc'], {
            height: 200,
            callbacks: ['onEpochEnd']
        });

        const start = performance.now();
        await currentModel.fit(trainXs, trainYs, {
            batchSize: 128,
            epochs: 5,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: callbacks
        });
        const duration = ((performance.now() - start)/1000).toFixed(2);
        log(`✅ training finished in ${duration}s`);
        // show final train/val metrics from last epoch? we could extract but fine
        metricsDisplay.innerText = `training completed (${duration}s) — check visor`;
    } catch (e) {
        log('❌ training error: ' + e.message, true);
    } finally {
        trainingStatus = 'idle';
        disposeTensors([trainXs, trainYs, valXs, valYs]);
    }
});

// --- Evaluate (overall + confusion + per‑class) ---
evaluateBtn.addEventListener('click', async () => {
    if (!testTensors || !currentModel) { alert('load model and test data'); return; }
    const { xs, ys } = testTensors;
    log('evaluating on test set...');
    try {
        const evalResult = currentModel.evaluate(xs, ys, { batchSize: 128 });
        const acc = evalResult[1].dataSync()[0]; // accuracy
        metricsDisplay.innerText = `test accuracy: ${(acc*100).toFixed(2)}%`;
        log(`test accuracy: ${(acc*100).toFixed(2)}%`);

        // predictions for confusion matrix & per‑class
        const preds = currentModel.predict(xs);
        const predLabels = preds.argMax(-1);
        const trueLabels = ys.argMax(-1);

        // confusion matrix using tfvis
        const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
        await tfvis.render.confusionMatrix(container, { values: await getConfusionArray(trueLabels, predLabels, 10) },
            { numClasses: 10, shadeDiagonal: true });

        // per‑class accuracy
        const perClassAcc = await computePerClassAccuracy(trueLabels, predLabels, 10);
        const perClassContainer = { name: 'Per‑Class Accuracy', tab: 'Evaluation' };
        await tfvis.render.barchart(perClassContainer, perClassAcc, { xLabel: 'class', yLabel: 'accuracy' });

        disposeTensors([preds, predLabels, trueLabels]);
    } catch (e) {
        log('❌ eval error: ' + e.message, true);
    }
});

// helper: compute confusion matrix counts as 2d array
async function getConfusionArray(trueIdx, predIdx, numClasses) {
    const trueVals = await trueIdx.data();
    const predVals = await predIdx.data();
    const matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    for (let i = 0; i < trueVals.length; i++) {
        matrix[trueVals[i]][predVals[i]]++;
    }
    return matrix;
}
async function computePerClassAccuracy(trueIdx, predIdx, numClasses) {
    const trueVals = await trueIdx.data();
    const predVals = await predIdx.data();
    const correct = Array(numClasses).fill(0);
    const total = Array(numClasses).fill(0);
    for (let i = 0; i < trueVals.length; i++) {
        total[trueVals[i]]++;
        if (trueVals[i] === predVals[i]) correct[trueVals[i]]++;
    }
    return total.map((t, i) => ({ index: i, value: t ? correct[i]/t : 0 }));
}

// --- Test 5 random preview ---
testFiveBtn.addEventListener('click', async () => {
    if (!testTensors) { alert('load test data first'); return; }
    const { xs, ys } = testTensors;
    const { images, labels } = getRandomTestBatch(xs, ys, 5);
    if (!images) return;

    previewContainer.innerHTML = ''; // clear
    for (let i = 0; i < images.length; i++) {
        const imgTensor = images[i]; // [28,28,1]
        const labelTensor = labels[i]; // [10]
        const trueIdx = labelTensor.argMax().dataSync()[0];

        // predict using current model if exists
        let predIdx = -1;
        if (currentModel) {
            const batchImg = imgTensor.expandDims(0); // [1,28,28,1]
            const pred = currentModel.predict(batchImg);
            predIdx = pred.argMax(-1).dataSync()[0];
            pred.dispose();
            batchImg.dispose();
        }

        const canvas = document.createElement('canvas');
        draw28x28ToCanvas(imgTensor, canvas, 2); // scale 2

        const itemDiv = document.createElement('div');
        itemDiv.className = 'preview-item';
        itemDiv.appendChild(canvas);
        const labelSpan = document.createElement('div');
        labelSpan.className = 'preview-label' + (predIdx === trueIdx ? ' correct' : ' wrong');
        labelSpan.innerText = `true: ${trueIdx} | pred: ${predIdx >= 0 ? predIdx : '?'}`;
        itemDiv.appendChild(labelSpan);
        previewContainer.appendChild(itemDiv);

        // dispose per‑image tensors after drawing (images, labels)
        imgTensor.dispose();
        labelTensor.dispose();
    }
});

// --- Save model (download) ---
saveModelBtn.addEventListener('click', async () => {
    if (!currentModel) { alert('no model'); return; }
    await currentModel.save('downloads://mnist-cnn');
    log('model download initiated');
});

// --- Load model from files (json + bin) ---
loadModelBtn.addEventListener('click', async () => {
    const jsonFile = modelJsonFile.files[0];
    const binFile = modelBinFile.files[0];
    if (!jsonFile || !binFile) { alert('select both model.json and .bin'); return; }
    try {
        const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
        if (currentModel) currentModel.dispose();
        currentModel = model;
        model.summary(null, null, (msg) => modelSummaryPre.innerText += msg + '\n');
        modelSummaryPre.innerText = 'loaded model:\n' + modelSummaryPre.innerText;
        log('model loaded from files');
    } catch (e) {
        log('❌ load model failed: ' + e.message, true);
    }
});

// --- Reset ---
resetBtn.addEventListener('click', resetState);

// --- Toggle visor ---
toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

// --- initial state ---
resetState();