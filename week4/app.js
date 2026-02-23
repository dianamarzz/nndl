// Global variables
let inputTensor, targetRamp;
let baselineModel, studentModel;
let optimizer;
let stepCount = 0;
let autoTrainInterval = null;
let currentArch = 'compression';

// Canvas contexts
const canvasInput = document.getElementById('canvasInput').getContext('2d');
const canvasBaseline = document.getElementById('canvasBaseline').getContext('2d');
const canvasStudent = document.getElementById('canvasStudent').getContext('2d');
const logArea = document.getElementById('logArea');

// Initialize everything when the page loads
async function init() {
    // Create fixed random input (16x16 with 1 channel)
    inputTensor = tf.tensor4d(
        Array(256).fill().map(() => Math.random() * 2 - 1), 
        [1, 16, 16, 1]
    );
    
    // Create target ramp pattern (increasing from top-left to bottom-right)
    const rampData = [];
    for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
            rampData.push((x / 15 + y / 15) / 2); // 0 to 1 ramp
        }
    }
    targetRamp = tf.tensor4d(rampData, [1, 16, 16, 1]);
    
    // Create models
    createModels(currentArch);
    
    // Render initial state
    renderAll();
    
    log('System initialized. Ready to train.');
}

function createModels(arch) {
    // Clean up old models if they exist
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    
    const inputShape = [16, 16, 1];
    
    // Baseline model (simple architecture for MSE baseline)
    baselineModel = tf.sequential({
        layers: [
            tf.layers.inputLayer({inputShape}),
            tf.layers.dense({units: 64, activation: 'relu'}),
            tf.layers.dense({units: 256, activation: 'sigmoid'}),
            tf.layers.reshape({targetShape: [16, 16, 1]})
        ]
    });
    
    // Student model with architecture based on selection
    studentModel = tf.sequential();
    studentModel.add(tf.layers.inputLayer({inputShape}));
    
    switch(arch) {
        case 'compression':
            // Compression: squeeze through smaller layer then expand
            studentModel.add(tf.layers.flatten());
            studentModel.add(tf.layers.dense({units: 32, activation: 'relu'})); // compressed
            studentModel.add(tf.layers.dense({units: 256, activation: 'sigmoid'}));
            studentModel.add(tf.layers.reshape({targetShape: [16, 16, 1]}));
            break;
            
        case 'transformation':
            // Transformation: maintain dimensions with conv-like dense layers
            studentModel.add(tf.layers.flatten());
            studentModel.add(tf.layers.dense({units: 128, activation: 'relu'}));
            studentModel.add(tf.layers.dense({units: 128, activation: 'relu'}));
            studentModel.add(tf.layers.dense({units: 256, activation: 'sigmoid'}));
            studentModel.add(tf.layers.reshape({targetShape: [16, 16, 1]}));
            break;
            
        case 'expansion':
            // Expansion: expand first, then process
            studentModel.add(tf.layers.flatten());
            studentModel.add(tf.layers.dense({units: 512, activation: 'relu'})); // expand
            studentModel.add(tf.layers.dense({units: 512, activation: 'relu'}));
            studentModel.add(tf.layers.dense({units: 256, activation: 'sigmoid'}));
            studentModel.add(tf.layers.reshape({targetShape: [16, 16, 1]}));
            break;
    }
    
    // Compile both models
    optimizer = tf.train.adam(0.01);
    
    baselineModel.compile({
        optimizer: optimizer,
        loss: 'meanSquaredError'
    });
    
    studentModel.compile({
        optimizer: optimizer,
        loss: studentLossFn  // Use your custom loss function
    });
    
    log(`Switched to ${arch} architecture`);
}

// Custom loss function (your provided code with small fix for targetRamp access)
const studentLossFn = (yTrue, yPred) => {
    // MSE loss - base reconstruction error
    const mseLoss = tf.losses.meanSquaredError(yTrue, yPred);
    
    // TV loss - using mean() for better scaling
    const rightDiff = yPred.slice([0,0,0,0], [-1,16,15,1]).sub(yPred.slice([0,0,1,0], [-1,16,15,1]));
    const downDiff = yPred.slice([0,0,0,0], [-1,15,16,1]).sub(yPred.slice([0,1,0,0], [-1,15,16,1]));
    const tvLoss = tf.square(rightDiff).mean().add(tf.square(downDiff).mean());
    
    // Direction loss - stronger influence to encourage ramp pattern
    const correlation = yPred.mul(targetRamp).mean();
    const dirLoss = correlation.neg();
    
    // New weights: TV=0.05, Direction=0.5
    return tf.tidy(() => {
        return mseLoss.add(tf.scalar(0.05).mul(tvLoss)).add(tf.scalar(0.5).mul(dirLoss));
    });
};

// Training step
async function trainStep() {
    stepCount++;
    
    // Train baseline (MSE only)
    const baselineHistory = await baselineModel.fit(inputTensor, targetRamp, {
        epochs: 1,
        verbose: 0
    });
    const baselineLoss = baselineHistory.history.loss[0];
    
    // Train student (custom loss)
    const studentHistory = await studentModel.fit(inputTensor, targetRamp, {
        epochs: 1,
        verbose: 0
    });
    const studentLoss = studentHistory.history.loss[0];
    
    // Update display
    renderAll();
    log(`Step ${stepCount} | Baseline loss: ${baselineLoss.toFixed(6)} | Student loss: ${studentLoss.toFixed(6)}`);
}

// Render all canvases
async function renderAll() {
    // Render input (scaled for visibility)
    const inputData = await inputTensor.squeeze().array();
    renderCanvas(canvasInput, inputData);
    
    // Render baseline prediction
    const baselinePred = baselineModel.predict(inputTensor);
    const baselineData = await baselinePred.squeeze().array();
    renderCanvas(canvasBaseline, baselineData);
    baselinePred.dispose();
    
    // Render student prediction
    const studentPred = studentModel.predict(inputTensor);
    const studentData = await studentPred.squeeze().array();
    renderCanvas(canvasStudent, studentData);
    studentPred.dispose();
}

// Helper to render a 16x16 array to canvas
function renderCanvas(ctx, data) {
    const canvas = ctx.canvas;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 16;
    tempCanvas.height = 16;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Create image data
    const imageData = tempCtx.createImageData(16, 16);
    for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
            const idx = (y * 16 + x) * 4;
            const val = Math.floor(Math.max(0, Math.min(255, (data[y][x] * 255))));
            imageData.data[idx] = val;     // R
            imageData.data[idx+1] = val;   // G
            imageData.data[idx+2] = val;   // B
            imageData.data[idx+3] = 255;   // A
        }
    }
    tempCtx.putImageData(imageData, 0, 0);
    
    // Scale up to canvas size
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, 16, 16, 0, 0, canvas.width, canvas.height);
}

// Logging function
function log(message) {
    logArea.innerHTML += message + '<br>';
    logArea.scrollTop = logArea.scrollHeight;
}

// Reset models to random weights
function reset() {
    stepCount = 0;
    createModels(currentArch);
    renderAll();
    log('Weights reset');
}

// Event listeners
document.getElementById('trainStep').addEventListener('click', () => {
    trainStep();
});

document.getElementById('autoTrain').addEventListener('click', (e) => {
    if (autoTrainInterval) {
        clearInterval(autoTrainInterval);
        autoTrainInterval = null;
        e.target.textContent = '▶ Auto Train';
    } else {
        autoTrainInterval = setInterval(() => trainStep(), 100);
        e.target.textContent = '⏸ Pause';
    }
});

document.getElementById('reset').addEventListener('click', reset);

// Architecture selector
document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.checked) {
            currentArch = e.target.value;
            reset(); // Reset and create new models
        }
    });
});

// Start everything when page loads
window.addEventListener('load', init);