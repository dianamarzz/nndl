/**
 * app.js
 * main UI wiring, model build, training, evaluation, file save/load.
 * Uses global tf, tfvis and functions from data-loader.js
 */
(function() {
    // ---------- DOM references ----------
    const trainFileInput = document.getElementById('train-file');
    const testFileInput = document.getElementById('test-file');
    const trainCountSpan = document.getElementById('train-count');
    const testCountSpan = document.getElementById('test-count');
    const dataStatus = document.getElementById('data-status');
    const trainingLog = document.getElementById('training-log');
    const metricsDisplay = document.getElementById('metrics-display');
    const modelInfoDiv = document.getElementById('model-info');
    const previewRow = document.getElementById('preview-row');

    // Buttons
    const loadDataBtn = document.getElementById('load-data-btn');
    const trainBtn = document.getElementById('train-btn');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const testFiveBtn = document.getElementById('test-five-btn');
    const saveModelBtn = document.getElementById('save-model-btn');
    const loadModelBtn = document.getElementById('load-model-btn');
    const resetBtn = document.getElementById('reset-btn');
    const toggleVisorBtn = document.getElementById('toggle-visor-btn');

    // Hidden model file inputs
    const modelJsonFile = document.getElementById('model-json-file');
    const modelBinFile = document.getElementById('model-bin-file');

    // ---------- App state (tensors & model) ----------
    let trainXs, trainYs, testXs, testYs;         // full train/test tensors
    let valXs, valYs;                             // validation split (created on load)
    let currentModel = null;

    // ---------- Helper: safe tensor disposal (except kept ones) ----------
    function disposeDatasets() {
        if (trainXs) { trainXs.dispose(); trainXs = null; }
        if (trainYs) { trainYs.dispose(); trainYs = null; }
        if (testXs) { testXs.dispose(); testXs = null; }
        if (testYs) { testYs.dispose(); testYs = null; }
        if (valXs) { valXs.dispose(); valXs = null; }
        if (valYs) { valYs.dispose(); valYs = null; }
    }

    // clear everything including model
    async function resetAll() {
        // dispose tensors
        disposeDatasets();
        
        // dispose model
        if (currentModel) {
            currentModel.dispose();
            currentModel = null;
        }
        
        // clear UI
        trainCountSpan.innerText = 'train: —';
        testCountSpan.innerText = 'test: —';
        dataStatus.innerText = 'reset done';
        trainingLog.innerText = '—';
        metricsDisplay.innerText = '—';
        modelInfoDiv.innerHTML = '⚙️ model info: none';
        previewRow.innerHTML = '<div class="preview-card">—</div>';
        
        // clear file inputs
        trainFileInput.value = '';
        testFileInput.value = '';
        modelJsonFile.value = '';
        modelBinFile.value = '';
        
        // close visor surfaces? optional
        dataStatus.innerText = 'all cleared';
    }

    // ---------- load data from both files ----------
    async function onLoadData() {
        try {
            dataStatus.innerText = 'reading files...';
            if (!trainFileInput.files[0] || !testFileInput.files[0]) {
                alert('please select both train and test CSV files');
                return;
            }

            // dispose old datasets
            disposeDatasets();

            // parse in parallel with loading indicators
            dataStatus.innerText = 'parsing train CSV...';
            const train = await window.loadTrainFromFiles(trainFileInput.files[0]);
            
            dataStatus.innerText = 'parsing test CSV...';
            const test = await window.loadTestFromFiles(testFileInput.files[0]);

            trainXs = train.xs;
            trainYs = train.ys;
            testXs = test.xs;
            testYs = test.ys;

            // create validation split (10%) - this now properly handles indices
            dataStatus.innerText = 'splitting validation set...';
            const split = window.splitTrainVal(trainXs, trainYs, 0.1);
            
            // replace train with split train, keep val
            trainXs.dispose(); trainYs.dispose(); // dispose old full train
            trainXs = split.trainXs;
            trainYs = split.trainYs;
            valXs = split.valXs;
            valYs = split.valYs;

            // update counts
            trainCountSpan.innerText = `train: ${trainXs.shape[0]}`;
            testCountSpan.innerText = `test: ${testXs.shape[0]}`;
            dataStatus.innerText = `✅ loaded: train ${trainXs.shape[0]}, val ${valXs.shape[0]}, test ${testXs.shape[0]}`;
            
            // optional: show sample counts
            trainingLog.innerText = `ready: ${trainXs.shape[0]} training + ${valXs.shape[0]} validation samples`;
        } catch (e) {
            console.error(e);
            dataStatus.innerText = `❌ error: ${e.message}`;
            alert(`Error loading data: ${e.message}`);
        }
    }

    // ---------- build model ----------
    function createModel() {
        if (currentModel) {
            currentModel.dispose();
        }
        
        const model = tf.sequential();
        
        // First Conv layer
        model.add(tf.layers.conv2d({
            filters: 32, 
            kernelSize: 3, 
            activation: 'relu', 
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        // Second Conv layer
        model.add(tf.layers.conv2d({ 
            filters: 64, 
            kernelSize: 3, 
            activation: 'relu', 
            padding: 'same' 
        }));
        
        // Pooling and dropout
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        
        // Dense layers
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        currentModel = model;
        
        // show summary in modelInfo
        modelInfoDiv.innerHTML = '⚙️ model built:<br><pre>';
        model.summary(undefined, (line) => {
            modelInfoDiv.innerHTML += line + '<br>';
        });
        modelInfoDiv.innerHTML += '</pre>';
        
        return model;
    }

    // ---------- train ----------
    async function onTrain() {
        if (!trainXs || !valXs) { 
            alert('load data first'); 
            return; 
        }
        
        if (!currentModel) {
            createModel();
        }

        trainingLog.innerText = 'training... (watch visor)';
        metricsDisplay.innerText = 'training in progress...';

        // tfvis callbacks
        const container = { name: 'MNIST Training', tab: 'Training' };
        const callbacks = tfvis.show.fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc'], {
            callbacks: ['onEpochEnd'],
            height: 300
        });

        const batchSize = 128;
        const epochs = 5;  // 5 for demo speed, can increase to 10

        const start = performance.now();
        try {
            await currentModel.fit(trainXs, trainYs, {
                batchSize,
                epochs,
                validationData: [valXs, valYs],
                callbacks,
                shuffle: true,
                verbose: 0
            });
            
            const duration = ((performance.now() - start) / 1000).toFixed(2);
            trainingLog.innerText = `✅ finished ${epochs} epochs in ${duration}s`;
            metricsDisplay.innerText = 'training done. click Evaluate for test accuracy.';
        } catch (e) {
            console.error(e);
            trainingLog.innerText = `❌ train error: ${e.message}`;
            metricsDisplay.innerText = 'training failed';
        }
    }

    // ---------- evaluation: accuracy + per‑class + confusion matrix ----------
    async function onEvaluate() {
        if (!testXs || !testYs || !currentModel) { 
            alert('load data and train/load a model first'); 
            return; 
        }
        
        metricsDisplay.innerText = 'evaluating...';
        
        try {
            // Overall accuracy
            const evalResult = currentModel.evaluate(testXs, testYs);
            const acc = evalResult[1].dataSync()[0];
            metricsDisplay.innerText = `test accuracy: ${(acc * 100).toFixed(2)}%`;
            
            // Get predictions for confusion matrix
            const preds = currentModel.predict(testXs).argMax(-1);
            const labels = testYs.argMax(-1);
            
            const predsArr = await preds.array();
            const labelsArr = await labels.array();
            
            // Clean up intermediate tensors
            preds.dispose();
            
            // Render confusion matrix
            tfvis.render.confusionMatrix(
                { name: 'Confusion Matrix', tab: 'Evaluation' },
                { values: labelsArr, predictions: predsArr },
                { 
                    numLabels: 10,
                    width: 400,
                    height: 400,
                    shadeDiagonal: true
                }
            );

            // Calculate per-class accuracy
            const perClass = Array(10).fill(0).map(() => ({ correct: 0, total: 0 }));
            for (let i = 0; i < labelsArr.length; i++) {
                const l = labelsArr[i];
                const p = predsArr[i];
                perClass[l].total++;
                if (l === p) perClass[l].correct++;
            }
            
            const accuracyPerClass = perClass.map((c, idx) => ({ 
                index: idx, 
                accuracy: c.total > 0 ? c.correct / c.total : 0 
            }));
            
            // Render bar chart
            tfvis.render.barchart(
                { name: 'Per-class Accuracy', tab: 'Evaluation' },
                accuracyPerClass.map(d => ({ 
                    value: d.accuracy, 
                    label: `Class ${d.index}` 
                })),
                { 
                    width: 400,
                    height: 300,
                    xLabel: 'Digit Class',
                    yLabel: 'Accuracy'
                }
            );
            
            trainingLog.innerText = 'evaluation complete - check visor';
        } catch (e) {
            console.error(e);
            metricsDisplay.innerText = `evaluation error: ${e.message}`;
        }
    }

    // ---------- test 5 random preview ----------
    async function onTestFive() {
        if (!testXs || !testYs) { 
            alert('load test data first'); 
            return; 
        }
        
        if (!currentModel) { 
            alert('train or load a model first'); 
            return; 
        }

        try {
            const batch = window.getRandomTestBatch(testXs, testYs, 5);
            if (!batch) return;
            
            const { xs: imgs, ys: labelsOneHot } = batch;
            const preds = currentModel.predict(imgs).argMax(-1);
            const labels = labelsOneHot.argMax(-1);

            const predsArr = await preds.array();
            const labelsArr = await labels.array();

            // clear preview row and create 5 cards
            previewRow.innerHTML = '';
            
            for (let i = 0; i < 5; i++) {
                const card = document.createElement('div');
                card.className = 'preview-card';

                const canvas = document.createElement('canvas');
                // Use draw function with scale 2 for 56x56 display
                const imgTensor = imgs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
                window.draw28x28ToCanvas(imgTensor, canvas, 2);
                imgTensor.dispose();

                const labelSpan = document.createElement('div');
                labelSpan.className = 'preview-label';
                const correct = predsArr[i] === labelsArr[i];
                labelSpan.innerHTML = `p:${predsArr[i]} / gt:${labelsArr[i]}`;
                labelSpan.classList.add(correct ? 'correct' : 'wrong');

                card.appendChild(canvas);
                card.appendChild(labelSpan);
                previewRow.appendChild(card);
            }

            // Clean up batch tensors
            imgs.dispose();
            labelsOneHot.dispose();
            preds.dispose();
            labels.dispose();
            
            trainingLog.innerText = 'random test samples displayed';
        } catch (e) {
            console.error(e);
            trainingLog.innerText = `preview error: ${e.message}`;
        }
    }

    // ---------- save model (download) ----------
    function onSaveModel() {
        if (!currentModel) { 
            alert('no model to save'); 
            return; 
        }
        
        currentModel.save('downloads://mnist-cnn').then(() => {
            trainingLog.innerText = '✅ model saved as mnist-cnn.json and weights.bin';
        }).catch(e => {
            trainingLog.innerText = `❌ save error: ${e.message}`;
        });
    }

    // ---------- load model from file inputs ----------
    async function onLoadModelFromFiles() {
        const jsonFile = modelJsonFile.files[0];
        const binFile = modelBinFile.files[0];
        
        if (!jsonFile || !binFile) {
            alert('select both model.json and weights.bin files');
            return;
        }
        
        try {
            trainingLog.innerText = 'loading model from files...';
            const model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
            
            // Dispose old model if exists
            if (currentModel) {
                currentModel.dispose();
            }
            
            currentModel = model;
            
            // Display model info
            modelInfoDiv.innerHTML = '⚙️ model loaded from files:<br><pre>';
            model.summary(undefined, (line) => {
                modelInfoDiv.innerHTML += line + '<br>';
            });
            modelInfoDiv.innerHTML += '</pre>';
            
            trainingLog.innerText = '✅ model reloaded successfully';
            dataStatus.innerText = 'model ready for inference';
        } catch (e) {
            console.error(e);
            trainingLog.innerText = `❌ load error: ${e.message}`;
            alert(`Failed to load model: ${e.message}`);
        }
    }

    // ---------- toggle visor ----------
    function onToggleVisor() {
        const visor = tfvis.visor();
        if (visor.isOpen()) {
            visor.close();
        } else {
            visor.open();
        }
    }

    // ---------- wire event listeners ----------
    loadDataBtn.addEventListener('click', onLoadData);
    trainBtn.addEventListener('click', onTrain);
    evaluateBtn.addEventListener('click', onEvaluate);
    testFiveBtn.addEventListener('click', onTestFive);
    saveModelBtn.addEventListener('click', onSaveModel);
    loadModelBtn.addEventListener('click', onLoadModelFromFiles);
    resetBtn.addEventListener('click', resetAll);
    toggleVisorBtn.addEventListener('click', onToggleVisor);

    // Initialize with a fresh model
    createModel();
    
    // Initial status
    dataStatus.innerText = 'ready - upload CSV files and click Load Data';
})();