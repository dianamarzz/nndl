// app.js – UI wiring, model, training, evaluation, file save/load, preview

(async function() {
    // ----- state -----
    let trainXs, trainYs, testXs, testYs;
    let currentModel = null;

    // DOM elements
    const trainFile = document.getElementById('train-file');
    const testFile = document.getElementById('test-file');
    const dataStatus = document.getElementById('data-status');
    const accStatus = document.getElementById('acc-status');
    const logPanel = document.getElementById('log-panel');
    const modelSummaryDiv = document.getElementById('model-summary');
    const previewContainer = document.getElementById('preview-container');
    const toggleVisorBtn = document.getElementById('toggle-visor');
    const loadDataBtn = document.getElementById('load-data');
    const trainBtn = document.getElementById('train-btn');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const test5Btn = document.getElementById('test5-btn');
    const saveModelBtn = document.getElementById('save-model');
    const loadModelBtn = document.getElementById('load-model-btn');
    const resetBtn = document.getElementById('reset-btn');
    const modelJsonInput = document.getElementById('model-json-input');

    // Helper: log to panel
    function log(message) {
        logPanel.innerText += message + '\n';
        logPanel.scrollTop = logPanel.scrollHeight;
    }
    function clearLog() { logPanel.innerText = ''; }

    // Safely dispose tensors
    function disposeDataTensors() {
        if (trainXs) { trainXs.dispose(); trainXs = null; }
        if (trainYs) { trainYs.dispose(); trainYs = null; }
        if (testXs) { testXs.dispose(); testXs = null; }
        if (testYs) { testYs.dispose(); testYs = null; }
    }

    // Reset everything
    async function resetAll() {
        disposeDataTensors();
        if (currentModel) {
            currentModel.dispose();
            currentModel = null;
        }
        // Reset preview
        previewContainer.innerHTML = '';
        for (let i = 0; i < 5; i++) {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = '—';
            previewContainer.appendChild(div);
        }
        dataStatus.innerText = '📦 no data';
        accStatus.innerText = 'accuracy: —';
        modelSummaryDiv.innerText = 'no model';
        clearLog();
        log('♻️ reset done');
    }

    // Create CNN model with proper initialization
    function createModel() {
        const model = tf.sequential();
        
        // First conv layer with He normal initialization for ReLU
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
            kernelInitializer: 'glorotNormal'  // Xavier init for softmax
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),  // Explicit learning rate
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    function logModelSummary() {
        if (!currentModel) return;
        const summary = [];
        currentModel.layers.forEach((l, i) => {
            summary.push(`${i}: ${l.name} | output: ${l.outputShape}`);
        });
        modelSummaryDiv.innerText = summary.join('\n') || 'model built';
    }

    // Load data handler
    loadDataBtn.addEventListener('click', async () => {
        try {
            if (!trainFile.files[0] || !testFile.files[0]) {
                alert('please select both train and test CSV files');
                return;
            }
            clearLog();
            log('📂 parsing train file...');
            const train = await window.dataLoader.loadTrainFromFiles(trainFile.files[0]);
            log('📂 parsing test file...');
            const test = await window.dataLoader.loadTestFromFiles(testFile.files[0]);

            disposeDataTensors();
            trainXs = train.xs; trainYs = train.ys;
            testXs = test.xs; testYs = test.ys;

            dataStatus.innerText = `📊 train: ${trainXs.shape[0]}  |  test: ${testXs.shape[0]}`;
            log(`✅ train samples: ${trainXs.shape[0]}, test: ${testXs.shape[0]}`);
            
            // Verify data range
            const sampleMin = trainXs.slice([0,0,0,0], [1,1,1,1]).dataSync()[0];
            const sampleMax = trainXs.slice([0,0,0,0], [1,1,1,1]).dataSync()[0];
            log(`📈 sample pixel range: [${sampleMin.toFixed(3)}, ${sampleMax.toFixed(3)}]`);
            
            if (!currentModel) {
                currentModel = createModel();
                logModelSummary();
            }
        } catch (e) {
            log(`❌ load error: ${e.message}`);
            console.error(e);
        }
    });

    // Training handler
    trainBtn.addEventListener('click', async () => {
        if (!trainXs || !trainYs) { alert('load data first'); return; }
        if (!currentModel) currentModel = createModel();

        try {
            const { trainXs: subTrainXs, trainYs: subTrainYs, valXs, valYs } =
                window.dataLoader.splitTrainVal(trainXs, trainYs, 0.1);

            log(`🏋️ training on ${subTrainXs.shape[0]} samples, val ${valXs.shape[0]}`);

            // tfvis callbacks
            const container = { name: 'MNIST Training', tab: 'Training' };
            const callbacks = tfvis.show.fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc'], {
                callbacks: ['onEpochEnd'],
                width: 400,
                height: 300
            });

            const start = performance.now();
            
            // Train with more epochs and early stopping
            await currentModel.fit(subTrainXs, subTrainYs, {
                batchSize: 128,
                epochs: 10,  // Increased epochs
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: callbacks,
                verbose: 1
            });
            
            const duration = ((performance.now() - start) / 1000).toFixed(1);
            log(`✅ training finished in ${duration}s`);

            // Quick evaluation on validation set
            const valEval = currentModel.evaluate(valXs, valYs);
            const valAcc = valEval[1].dataSync()[0];
            log(`📊 validation accuracy: ${(valAcc * 100).toFixed(2)}%`);

            tf.dispose([subTrainXs, subTrainYs, valXs, valYs]);
        } catch (err) {
            log(`❌ train error: ${err.message}`);
            console.error(err);
        }
    });

    // Evaluation handler
    evaluateBtn.addEventListener('click', async () => {
        if (!testXs || !testYs || !currentModel) {
            alert('need model and test data');
            return;
        }
        try {
            log('🔍 evaluating on test set...');
            
            // Evaluate on test set
            const evalResult = currentModel.evaluate(testXs, testYs, { batchSize: 128 });
            const loss = evalResult[0].dataSync()[0];
            const acc = evalResult[1].dataSync()[0];
            
            accStatus.innerText = `accuracy: ${(acc * 100).toFixed(2)}%`;
            log(`test loss = ${loss.toFixed(4)}, accuracy = ${(acc * 100).toFixed(2)}%`);

            // Get predictions for confusion matrix
            const preds = currentModel.predict(testXs);
            const trueLabels = testYs.argMax(-1);
            const predLabels = preds.argMax(-1);

            // Calculate per-class accuracy
            const equality = trueLabels.equal(predLabels).cast('float32');
            const trueOneHot = tf.oneHot(trueLabels, 10);
            const classMask = trueOneHot.mul(equality.expandDims(1));
            const classCorrect = classMask.sum(0);
            const classCounts = trueOneHot.sum(0);
            const classAcc = classCorrect.div(classCounts);
            const classAccArray = await classAcc.array();

            // Log per-class accuracy
            log('\n📊 Per-class accuracy:');
            classAccArray.forEach((acc, i) => {
                log(`  class ${i}: ${(acc * 100).toFixed(1)}%`);
            });

            // Render bar chart
            tfvis.render.barchart(
                { name: 'Per‑class accuracy', tab: 'Evaluation' },
                classAccArray.map((v, i) => ({ index: i, accuracy: v })),
                { xLabel: 'digit', yLabel: 'accuracy', width: 400 }
            );

            // Confusion matrix
            const confusionMatrix = await tf.math.confusionMatrix(trueLabels, predLabels, 10);
            tfvis.render.confusionMatrix(
                { name: 'Confusion matrix', tab: 'Evaluation' },
                confusionMatrix,
                { shadeDiagonal: true, width: 400 }
            );

            // Clean up
            tf.dispose([preds, trueLabels, predLabels, equality, trueOneHot, 
                       classMask, classCorrect, classCounts, classAcc, confusionMatrix]);

        } catch (err) {
            log(`❌ eval error: ${err.message}`);
            console.error(err);
        }
    });

    // Test 5 random preview
    test5Btn.addEventListener('click', async () => {
        if (!testXs || !testYs || !currentModel) {
            alert('load test data and model first');
            return;
        }
        try {
            const { batchXs, batchYs } = window.dataLoader.getRandomTestBatch(testXs, testYs, 5);
            const predProbs = currentModel.predict(batchXs);
            const predLabels = predProbs.argMax(-1);
            const trueLabels = batchYs.argMax(-1);

            const predArr = await predLabels.array();
            const trueArr = await trueLabels.array();

            // Clear and rebuild preview
            previewContainer.innerHTML = '';
            for (let i = 0; i < 5; i++) {
                const div = document.createElement('div');
                div.className = 'preview-item';

                const canvas = document.createElement('canvas');
                canvas.width = 112; canvas.height = 112;
                canvas.style.width = '84px'; canvas.style.height = '84px';

                const imgTensor = batchXs.slice([i,0,0,0], [1,28,28,1]).squeeze();
                window.dataLoader.draw28x28ToCanvas(imgTensor, canvas, 4);

                const labelSpan = document.createElement('div');
                labelSpan.className = `pred-label ${predArr[i] === trueArr[i] ? 'correct' : 'wrong'}`;
                labelSpan.innerText = `${predArr[i]}`;

                div.appendChild(canvas);
                div.appendChild(labelSpan);
                previewContainer.appendChild(div);
            }
            
            tf.dispose([batchXs, batchYs, predProbs, predLabels, trueLabels]);
        } catch (err) {
            log(`❌ preview error: ${err.message}`);
        }
    });

    // Save model
    saveModelBtn.addEventListener('click', async () => {
        if (!currentModel) { alert('no model to save'); return; }
        try {
            await currentModel.save('downloads://mnist-cnn');
            log('💾 model saved as mnist-cnn');
        } catch (e) { 
            log('save error: ' + e); 
        }
    });

    // Load model from files
    loadModelBtn.addEventListener('click', () => modelJsonInput.click());
    
    modelJsonInput.addEventListener('change', async (event) => {
        const files = event.target.files;
        if (!files || files.length < 2) {
            alert('please select both model.json and .bin file');
            return;
        }
        try {
            const model = await tf.loadLayersModel(tf.io.browserFiles(files));
            if (currentModel) currentModel.dispose();
            currentModel = model;
            // Recompile to ensure optimizer is set
            currentModel.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            logModelSummary();
            log('✅ model loaded from files');
        } catch (e) {
            log(`❌ load model error: ${e.message}`);
        }
        modelJsonInput.value = '';
    });

    // Reset
    resetBtn.addEventListener('click', resetAll);

    // Toggle tfvis visor
    toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

    // Initial log
    log('🚀 MNIST trainer ready. Upload CSV files and click Load data.');
})();