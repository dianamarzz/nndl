// app.js – main UI controller, training, evaluation, model I/O

import { 
    loadTrainFromFiles, 
    loadTestFromFiles, 
    splitTrainVal, 
    getRandomTestBatch, 
    draw28x28ToCanvas 
} from './data-loader.js';

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
    const timestamp = new Date().toLocaleTimeString();
    logPanel.innerText += `\n[${timestamp}] ${msg}`;
    console.log(msg);
    // Auto-scroll to bottom
    logPanel.scrollTop = logPanel.scrollHeight;
}

function clearLog() { 
    logPanel.innerText = '[ready]'; 
}

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
        Object.values(valSplit).forEach(t => {
            if (t && t.dispose) t.dispose();
        });
        valSplit = null;
    }
}

// Reset everything
async function resetAll() {
    log('🔄 Resetting application...');
    if (currentModel) {
        currentModel.dispose();
        currentModel = null;
    }
    disposeTensors();
    tf.disposeVariables(); // extra cleanup
    dataStatus.innerText = '📌 reset: no data';
    metricsDisplay.innerText = 'accuracy: —';
    previewRow.innerHTML = '<div class="preview-item">⬜ no images</div>';
    modelSummaryText.innerText = '— no model —';
    tfvis.visor().close(); // optional close
    log('✅ Reset complete');
}

// Build fresh CNN model
function createModel() {
    log('🔧 Creating new CNN model...');
    tf.engine().startScope(); // helps with dispose
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same', 
        inputShape: [28, 28, 1] 
    }));
    model.add(tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same' 
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy', 
        metrics: ['accuracy'] 
    });
    tf.engine().endScope();
    log('✅ Model created');
    return model;
}

// Initialize model on page load
currentModel = createModel();
updateModelInfo();
clearLog();

// ---------- event handlers ----------
loadDataBtn.addEventListener('click', async () => {
    try {
        if (!trainFile.files[0] || !testFile.files[0]) {
            alert('Please select both train and test CSV files');
            return;
        }
        
        log('📂 Loading data...');
        await resetAll(); // start fresh
        
        log('📂 Loading training data...');
        trainTensors = await loadTrainFromFiles(trainFile.files[0]);
        log(`✅ Train samples: ${trainTensors.xs.shape[0]}`);
        
        log('📂 Loading test data...');
        testTensors = await loadTestFromFiles(testFile.files[0]);
        log(`✅ Test samples: ${testTensors.xs.shape[0]}`);
        
        dataStatus.innerText = `📊 train: ${trainTensors.xs.shape[0]} | test: ${testTensors.xs.shape[0]}`;

        // Create validation split (10%)
        log('✂️ Creating validation split...');
        valSplit = splitTrainVal(trainTensors.xs, trainTensors.ys, 0.1);
        log(`Validation split: ${valSplit.trainXs.shape[0]} train / ${valSplit.valXs.shape[0]} val`);

        if (!currentModel) {
            currentModel = createModel();
        }
        updateModelInfo();
        log('✅ Data loading complete');
    } catch (err) {
        log('❌ Load error: ' + err.message);
        console.error(err);
    }
});

trainBtn.addEventListener('click', async () => {
    if (!valSplit || !currentModel) {
        alert('Load data first');
        return;
    }
    
    try {
        log('🏋️ Training started (epochs=5, batch=128)');
        const { trainXs, trainYs, valXs, valYs } = valSplit;
        
        // Create containers for tfvis
        const container = { name: 'MNIST Training', tab: 'training' };
        const callbacks = tfvis.show.fitCallbacks(container, 
            ['loss', 'val_loss', 'acc', 'val_acc'], 
            {
                callbacks: ['onEpochEnd'],
                height: 300
            }
        );

        const start = performance.now();
        
        await currentModel.fit(trainXs, trainYs, {
            batchSize: 128,
            epochs: 5,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: callbacks,
            verbose: 0
        });
        
        const duration = ((performance.now() - start) / 1000).toFixed(2);
        log(`✅ Training finished in ${duration}s`);
        updateModelInfo();
        
        // Show visor
        tfvis.visor().open();
    } catch (err) {
        log('❌ Train error: ' + err.message);
        console.error(err);
    }
});

evaluateBtn.addEventListener('click', async () => {
    if (!testTensors || !currentModel) {
        alert('Load data and train (or load model) first');
        return;
    }
    
    try {
        log('📊 Evaluating on test set...');
        const { xs, ys } = testTensors;
        
        const evalResult = await currentModel.evaluate(xs, ys, { batchSize: 128 });
        const acc = evalResult[1].dataSync()[0];
        const accPercent = (acc * 100).toFixed(2);
        metricsDisplay.innerText = `accuracy: ${accPercent}%`;
        log(`✅ Test accuracy: ${accPercent}%`);

        // Get predictions for confusion matrix
        const preds = currentModel.predict(xs);
        const trueLabels = tf.argMax(ys, 1).dataSync();
        const predLabels = tf.argMax(preds, 1).dataSync();
        
        // Calculate per-class accuracy
        const classAccuracy = Array(10).fill(0).map(() => ({ correct: 0, total: 0 }));
        for (let i = 0; i < trueLabels.length; i++) {
            const t = trueLabels[i];
            const p = predLabels[i];
            classAccuracy[t].total++;
            if (t === p) classAccuracy[t].correct++;
        }
        
        // Build confusion matrix
        const matrix = Array(10).fill(0).map(() => Array(10).fill(0));
        for (let i = 0; i < trueLabels.length; i++) {
            matrix[trueLabels[i]][predLabels[i]]++;
        }

        // Render visualizations
        tfvis.render.confusionMatrix(
            { name: 'Confusion Matrix', tab: 'evaluation' },
            { values: matrix },
            { height: 400 }
        );
        
        const perClassAcc = classAccuracy.map(c => c.correct / (c.total || 1));
        tfvis.render.barchart(
            { name: 'Per‑class Accuracy', tab: 'evaluation' },
            perClassAcc.map((acc, idx) => ({ index: idx, accuracy: acc })),
            { 
                xLabel: 'Digit Class', 
                yLabel: 'Accuracy',
                height: 300 
            }
        );

        // Cleanup
        preds.dispose();
        
        log('✅ Evaluation complete, check visor');
        tfvis.visor().setActiveTab('evaluation');
        tfvis.visor().open();
    } catch (err) {
        log('❌ Evaluation error: ' + err.message);
        console.error(err);
    }
});

testFiveBtn.addEventListener('click', async () => {
    if (!testTensors || !currentModel) {
        alert('Load test data and model first');
        return;
    }
    
    try {
        log('🎲 Generating random test preview...');
        const batch = getRandomTestBatch(testTensors.xs, testTensors.ys, 5);
        const predProbs = currentModel.predict(batch.xs);
        const predLabels = tf.argMax(predProbs, 1).dataSync();
        const trueLabels = tf.argMax(batch.ys, 1).dataSync();

        // Clear preview row
        previewRow.innerHTML = '';

        // Create preview items
        for (let i = 0; i < 5; i++) {
            const canvas = document.createElement('canvas');
            const imgTensor = batch.xs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
            draw28x28ToCanvas(imgTensor, canvas, 4);
            imgTensor.dispose();

            const badge = document.createElement('span');
            badge.className = 'pred-badge ' + (predLabels[i] === trueLabels[i] ? 'pred-correct' : 'pred-wrong');
            badge.innerText = `${trueLabels[i]} → ${predLabels[i]}`;

            const item = document.createElement('div');
            item.className = 'preview-item';
            item.appendChild(canvas);
            item.appendChild(badge);
            previewRow.appendChild(item);
        }
        
        // Cleanup
        predProbs.dispose();
        batch.xs.dispose();
        batch.ys.dispose();
        
        log('✅ Preview updated');
    } catch (err) {
        log('❌ Preview error: ' + err.message);
        console.error(err);
    }
});

saveModelBtn.addEventListener('click', async () => {
    if (!currentModel) { 
        alert('No model to save'); 
        return; 
    }
    
    try {
        log('💾 Saving model...');
        await currentModel.save('downloads://mnist-cnn');
        log('✅ Model saved successfully');
    } catch (err) {
        log('❌ Save error: ' + err.message);
        console.error(err);
    }
});

loadModelBtn.addEventListener('click', () => {
    // Trigger file selection
    modelJsonFile.click();
});

modelJsonFile.addEventListener('change', () => {
    if (modelJsonFile.files[0]) {
        // Now ask for weights file
        modelWeightsFile.click();
    }
});

modelWeightsFile.addEventListener('change', async () => {
    if (modelJsonFile.files[0] && modelWeightsFile.files[0]) {
        try {
            log('📂 Loading model from files...');
            
            const model = await tf.loadLayersModel(tf.io.browserFiles([
                modelJsonFile.files[0],
                modelWeightsFile.files[0]
            ]));
            
            // Dispose old model and set new one
            if (currentModel) currentModel.dispose();
            currentModel = model;
            
            // Recompile to ensure optimizer is set
            currentModel.compile({ 
                optimizer: 'adam', 
                loss: 'categoricalCrossentropy', 
                metrics: ['accuracy'] 
            });
            
            updateModelInfo();
            log('✅ Model loaded successfully');
            
            // Clear file inputs for next load
            modelJsonFile.value = '';
            modelWeightsFile.value = '';
        } catch (err) {
            log('❌ Load model error: ' + err.message);
            console.error(err);
        }
    }
});

resetBtn.addEventListener('click', resetAll);

toggleVisorBtn.addEventListener('click', () => {
    const visor = tfvis.visor();
    if (visor.isOpen()) {
        visor.close();
        log('👁️ Visor closed');
    } else {
        visor.open();
        log('👁️ Visor opened');
    }
});