// app.js – main UI controller, training, evaluation, model I/O

import { loadTrainFromFiles, loadTestFromFiles, splitTrainVal, getRandomTestBatch, draw28x28ToCanvas } from './data-loader.js';

// DOM elements
const trainFile = document.getElementById('trainFile');
const testFile = document.getElementById('testFile');
const loadDataBtn = document.getElementById('loadDataBtn');
const trainBtn = document.getElementById('trainBtn');
const evaluateBtn = document.getElementById('evaluateBtn');
const testFiveBtn = document.getElementById('testFiveBtn');
const saveModelBtn = document.getElementById('saveModelBtn');
const loadModelBtn = document.getElementById('loadModelBtn');
const resetBtn = document.getElementById('resetBtn');
const toggleVisorBtn = document.getElementById('toggleVisorBtn');
const dataStatus = document.getElementById('dataStatus');
const metricsDisplay = document.getElementById('metricsDisplay');
const logPanel = document.getElementById('logPanel');
const previewRow = document.getElementById('previewRow');
const modelSummaryText = document.getElementById('modelSummaryText');
const modelJsonFile = document.getElementById('modelJsonFile');
const modelWeightsFile = document.getElementById('modelWeightsFile');

// Global state
let trainTensors = null;   // {xs, ys}
let testTensors = null;    // {xs, ys}
let currentModel = null;
let valSplit = null;       // cached for training (optional)

// helper log
function log(msg) {
    logPanel.innerText += '\n> ' + msg;
    console.log(msg);
}
function clearLog() { logPanel.innerText = '[ready]'; }

// Update UI with model summary (layers, params)
function updateModelInfo() {
    if (!currentModel) {
        modelSummaryText.innerText = '— no model —';
        return;
    }
    const layers = currentModel.layers.map(l => l.name + ' (' + l.outputShape.join('x') + ')').join(', ');
    const params = currentModel.countParams();
    modelSummaryText.innerText = `layers: ${layers} | params: ${params.toLocaleString()}`;
}

// Dispose tensors safely
function disposeTensors() {
    if (trainTensors) {
        trainTensors.xs.dispose();
        trainTensors.ys.dispose();
        trainTensors = null;
    }
    if (testTensors) {
        testTensors.xs.dispose();
        testTensors.ys.dispose();
        testTensors = null;
    }
    if (valSplit) {
        Object.values(valSplit).forEach(t => t?.dispose());
        valSplit = null;
    }
}

// Reset everything
async function resetAll() {
    if (currentModel) {
        currentModel.dispose();
        currentModel = null;
    }
    disposeTensors();
    tf.disposeVariables(); // extra cleanup
    dataStatus.innerText = '📌 reset: no data';
    metricsDisplay.innerText = 'accuracy: —';
    clearLog();
    previewRow.innerHTML = '<div class="preview-item">⬜ no images</div>';
    modelSummaryText.innerText = '— no model —';
    tfvis.visor().close(); // optional close
}

// Build fresh CNN model
function createModel() {
    tf.engine().startScope(); // helps with dispose
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
    tf.engine().endScope();
    return model;
}

// ---------- event handlers ----------
loadDataBtn.addEventListener('click', async () => {
    try {
        if (!trainFile.files[0] || !testFile.files[0]) {
            alert('Please select both train and test CSV files');
            return;
        }
        resetAll(); // start fresh
        log('📂 loading train data...');
        trainTensors = await loadTrainFromFiles(trainFile.files[0]);
        log(`✅ train samples: ${trainTensors.xs.shape[0]}`);
        log('📂 loading test data...');
        testTensors = await loadTestFromFiles(testFile.files[0]);
        log(`✅ test samples: ${testTensors.xs.shape[0]}`);
        dataStatus.innerText = `📊 train: ${trainTensors.xs.shape[0]} | test: ${testTensors.xs.shape[0]}`;

        // optional validation split (10%)
        valSplit = splitTrainVal(trainTensors.xs, trainTensors.ys, 0.1);
        log(`validation split: ${valSplit.trainXs.shape[0]} train / ${valSplit.valXs.shape[0]} val`);

        if (!currentModel) {
            currentModel = createModel();
            updateModelInfo();
        }
    } catch (err) {
        log('❌ load error: ' + err.message);
        console.error(err);
    }
});

trainBtn.addEventListener('click', async () => {
    if (!valSplit || !currentModel) {
        alert('Load data first');
        return;
    }
    try {
        log('🏋️ training started (epochs=5, batch=128)');
        const { trainXs, trainYs, valXs, valYs } = valSplit;
        const container = { name: 'MNIST Training', tab: 'training' };
        const callbacks = tfvis.show.fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc'], {
            callbacks: ['onEpochEnd'],
            height: 300
        });

        const start = performance.now();
        await currentModel.fit(trainXs, trainYs, {
            batchSize: 128,
            epochs: 5,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: callbacks,
            verbose: 0
        });
        const duration = ((performance.now() - start)/1000).toFixed(2);
        log(`✅ training finished in ${duration}s`);
        updateModelInfo();
    } catch (err) {
        log('❌ train error: ' + err.message);
    }
});

evaluateBtn.addEventListener('click', async () => {
    if (!testTensors || !currentModel) {
        alert('Load data and train (or load model) first');
        return;
    }
    try {
        log('📊 evaluating on test set...');
        const { xs, ys } = testTensors;
        const evalResult = await currentModel.evaluate(xs, ys, { batchSize: 128 });
        const acc = evalResult[1].dataSync()[0];
        metricsDisplay.innerText = `accuracy: ${(acc*100).toFixed(2)}%`;

        // confusion matrix and per‑class acc via tfvis
        const preds = currentModel.predict(xs);
        const trueLabels = tf.argMax(ys, 1).dataSync();
        const predLabels = tf.argMax(preds, 1).dataSync();
        preds.dispose();

        const classAccuracy = Array(10).fill(0).map(() => ({ correct:0, total:0 }));
        for (let i = 0; i < trueLabels.length; i++) {
            const t = trueLabels[i];
            const p = predLabels[i];
            classAccuracy[t].total++;
            if (t === p) classAccuracy[t].correct++;
        }
        const perClassAcc = classAccuracy.map(c => c.correct / (c.total || 1));

        tfvis.render.confusionMatrix({ name: 'confusion matrix', tab: 'evaluation' }, {
            values: await buildConfusionMatrix(trueLabels, predLabels, 10)
        });
        tfvis.render.barchart({ name: 'per‑class accuracy', tab: 'evaluation' }, {
            values: perClassAcc.map((acc, idx) => ({ index: idx, accuracy: acc }))
        }, { xLabel: 'class', yLabel: 'accuracy' });

        log('✅ evaluation complete, check visor');
        tfvis.visor().setActiveTab('evaluation');
    } catch (err) {
        log('❌ eval error: ' + err.message);
    }
});

async function buildConfusionMatrix(trueIds, predIds, numClasses) {
    const matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    for (let i=0; i<trueIds.length; i++) {
        matrix[trueIds[i]][predIds[i]]++;
    }
    return matrix;
}

testFiveBtn.addEventListener('click', async () => {
    if (!testTensors || !currentModel) {
        alert('Load test data and model first');
        return;
    }
    try {
        const batch = getRandomTestBatch(testTensors.xs, testTensors.ys, 5);
        const predProbs = currentModel.predict(batch.xs);
        const predLabels = tf.argMax(predProbs, 1).dataSync();
        const trueLabels = tf.argMax(batch.ys, 1).dataSync();

        // clear preview row
        previewRow.innerHTML = '';

        for (let i = 0; i < 5; i++) {
            const canvas = document.createElement('canvas');
            const imgTensor = batch.xs.slice([i,0,0,0], [1,28,28,1]);
            draw28x28ToCanvas(imgTensor, canvas, 4);
            imgTensor.dispose();

            const badge = document.createElement('span');
            badge.className = 'pred-badge ' + (predLabels[i] === trueLabels[i] ? 'pred-correct' : 'pred-wrong');
            badge.innerText = `true:${trueLabels[i]} pred:${predLabels[i]}`;

            const item = document.createElement('div');
            item.className = 'preview-item';
            item.appendChild(canvas);
            item.appendChild(badge);
            previewRow.appendChild(item);
        }
        predProbs.dispose();
        batch.xs.dispose(); batch.ys.dispose();
        log('🎲 random test preview updated');
    } catch (err) {
        log('❌ preview error: ' + err.message);
    }
});

saveModelBtn.addEventListener('click', async () => {
    if (!currentModel) { alert('No model to save'); return; }
    try {
        await currentModel.save('downloads://mnist-cnn');
        log('💾 model saved (downloads)');
    } catch (err) {
        log('❌ save error: ' + err.message);
    }
});

loadModelBtn.addEventListener('click', () => {
    // programmatically click hidden file inputs
    modelJsonFile.click();
    modelJsonFile.onchange = () => {
        if (modelJsonFile.files[0]) {
            modelWeightsFile.click();
        }
    };
    modelWeightsFile.onchange = async () => {
        if (modelJsonFile.files[0] && modelWeightsFile.files[0]) {
            try {
                const model = await tf.loadLayersModel(tf.io.browserFiles([
                    modelJsonFile.files[0],
                    modelWeightsFile.files[0]
                ]));
                if (currentModel) currentModel.dispose();
                currentModel = model;
                currentModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
                updateModelInfo();
                log('✅ model loaded from files');
            } catch (err) {
                log('❌ load model error: ' + err.message);
            }
        }
    };
});

resetBtn.addEventListener('click', resetAll);

toggleVisorBtn.addEventListener('click', () => {
    const visor = tfvis.visor();
    visor.isOpen() ? visor.close() : visor.open();
});

// init: create model placeholder (optional)
currentModel = createModel();
updateModelInfo();
clearLog();