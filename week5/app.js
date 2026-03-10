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
let trainTensors = null;
let testTensors = null;
let currentModel = null;
let valSplit = null;

// helper log
function log(msg) {
    const timestamp = new Date().toLocaleTimeString();
    logPanel.innerText += `\n[${timestamp}] ${msg}`;
    console.log(msg);
    logPanel.scrollTop = logPanel.scrollHeight;
}

function clearLog() { 
    logPanel.innerText = '[ready]'; 
}

// Update UI with model summary
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
    tf.disposeVariables();
    dataStatus.innerText = '📌 reset: no data';
    metricsDisplay.innerText = 'accuracy: —';
    previewRow.innerHTML = '<div class="preview-item">⬜ no images</div>';
    modelSummaryText.innerText = '— no model —';
    tfvis.visor().close();
    log('✅ Reset complete');
}

// Build fresh CNN model with better initialization
function createModel() {
    log('🔧 Creating new CNN model...');
    
    const model = tf.sequential();
    
    // First conv layer with He normal initialization
    model.add(tf.layers.conv2d({ 
        filters: 32, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same', 
        inputShape: [28, 28, 1],
        kernelInitializer: 'heNormal'
    }));
    
    model.add(tf.layers.conv2d({ 
        filters: 64, 
        kernelSize: 3, 
        activation: 'relu', 
        padding: 'same',
        kernelInitializer: 'heNormal'
    }));
    
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.flatten());
    
    model.add(tf.layers.dense({ 
        units: 128, 
        activation: 'relu',
        kernelInitializer: 'heNormal'
    }));
    
    model.add(tf.layers.dropout({ rate: 0.5 }));
    
    model.add(tf.layers.dense({ 
        units: 10, 
        activation: 'softmax',
        kernelInitializer: 'glorotUniform'
    }));
    
    // Use Adam with default learning rate
    const optimizer = tf.train.adam(0.001);
    
    model.compile({ 
        optimizer: optimizer, 
        loss: 'categoricalCrossentropy', 
        metrics: ['accuracy'] 
    });
    
    log('✅ Model created with He initialization');
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
        await resetAll();
        
        log('📂 Loading training data...');
        trainTensors = await loadTrainFromFiles(trainFile.files[0]);
        log(`✅ Train samples: ${trainTensors.xs.shape[0]}`);
        
        // Verify data range
        const sampleMin = trainTensors.xs.slice([0,0,0,0], [1,1,1,1]).dataSync()[0];
        const sampleMax = trainTensors.xs.slice([0,27,27,0], [1,1,1,1]).dataSync()[0];
        log(`📊 Pixel range: [${sampleMin.toFixed(3)}, ${sampleMax.toFixed(3)}]`);
        
        log('📂 Loading test data...');
        testTensors = await loadTestFromFiles(testFile.files[0]);
        log(`✅ Test samples: ${testTensors.xs.shape[0]}`);
        
        dataStatus.innerText = `📊 train: ${trainTensors.xs.shape[0]} | test: ${testTensors.xs.shape[0]}`;

        // Create validation split
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
        
        const history = await currentModel.fit(trainXs, trainYs, {
            batchSize: 128,
            epochs: 5,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: callbacks,
            verbose: 1  // Set to 1 to see progress in console
        });
        
        const duration = ((performance.now() - start) / 1000).toFixed(2);
        
        // Log final training accuracy
        const finalTrainAcc = history.history.acc[history.history.acc.length - 1];
        const finalValAcc = history.history.val_acc[history.history.val_acc.length - 1];
        log(`✅ Training finished in ${duration}s`);
        log(`📈 Final train accuracy: ${(finalTrainAcc * 100).toFixed(2)}%`);
        log(`📊 Final validation accuracy: ${(finalValAcc * 100).toFixed(2)}%`);
        
        updateModelInfo();
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
        
        // Get predictions first
        const preds = currentModel.predict(xs);
        const predLabels = tf.argMax(preds, 1);
        const trueLabels = tf.argMax(ys, 1);
        
        // Calculate accuracy manually to verify
        const correct = tf.equal(predLabels, trueLabels);
        const accuracy = tf.mean(tf.cast(correct, 'float32'));
        const accValue = await accuracy.dataSync()[0];
        
        metricsDisplay.innerText = `accuracy: ${(accValue * 100).toFixed(2)}%`;
        log(`✅ Test accuracy: ${(accValue * 100).toFixed(2)}%`);

        // Get data for confusion matrix
        const predLabelsArray = await predLabels.array();
        const trueLabelsArray = await trueLabels.array();
        
        // Build confusion matrix
        const matrix = Array(10).fill(0).map(() => Array(10).fill(0));
        for (let i = 0; i < trueLabelsArray.length; i++) {
            matrix[trueLabelsArray[i]][predLabelsArray[i]]++;
        }

        // Calculate per-class accuracy
        const perClassAcc = [];
        for (let i = 0; i < 10; i++) {
            const total = matrix[i].reduce((a, b) => a + b, 0);
            const correct = matrix[i][i];
            perClassAcc.push({
                index: i,
                accuracy: total > 0 ? correct / total : 0
            });
        }

        // Render visualizations
        tfvis.render.confusionMatrix(
            { name: 'Confusion Matrix', tab: 'evaluation' },
            { values: matrix },
            { 
                height: 400,
                colorMap: 'blues'
            }
        );
        
        tfvis.render.barchart(
            { name: 'Per‑class Accuracy', tab: 'evaluation' },
            perClassAcc,
            { 
                xLabel: 'Digit Class', 
                yLabel: 'Accuracy',
                height: 300
            }
        );

        // Cleanup
        preds.dispose();
        predLabels.dispose();
        trueLabels.dispose();
        correct.dispose();
        accuracy.dispose();
        
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
        const predLabels = tf.argMax(predProbs, 1);
        const trueLabels = tf.argMax(batch.ys, 1);
        
        const predArray = await predLabels.array();
        const trueArray = await trueLabels.array();

        // Clear preview row
        previewRow.innerHTML = '';

        // Create preview items
        for (let i = 0; i < 5; i++) {
            const canvas = document.createElement('canvas');
            const imgTensor = batch.xs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
            await draw28x28ToCanvas(imgTensor, canvas, 4);
            imgTensor.dispose();

            const badge = document.createElement('span');
            badge.className = 'pred-badge ' + (predArray[i] === trueArray[i] ? 'pred-correct' : 'pred-wrong');
            badge.innerText = `${trueArray[i]} → ${predArray[i]}`;

            const item = document.createElement('div');
            item.className = 'preview-item';
            item.appendChild(canvas);
            item.appendChild(badge);
            previewRow.appendChild(item);
        }
        
        // Cleanup
        predProbs.dispose();
        predLabels.dispose();
        trueLabels.dispose();
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
    modelJsonFile.click();
});

modelJsonFile.addEventListener('change', () => {
    if (modelJsonFile.files[0]) {
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
            
            if (currentModel) currentModel.dispose();
            currentModel = model;
            
            currentModel.compile({ 
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy', 
                metrics: ['accuracy'] 
            });
            
            updateModelInfo();
            log('✅ Model loaded successfully');
            
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