// app.js – UI wiring, model, training, evaluation, file save/load, preview

(async function() {
    // ----- state -----
    let trainXs, trainYs, testXs, testYs;          // full tensors
    let currentModel = null;
    let trainingStatus = 'idle';

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

    // helper: log to panel
    function log(message) {
        logPanel.innerText += message + '\n';
        logPanel.scrollTop = logPanel.scrollHeight;
    }
    function clearLog() { logPanel.innerText = ''; }

    // safely dispose tensors (if exist)
    function disposeDataTensors() {
        if (trainXs) { trainXs.dispose(); trainXs = null; }
        if (trainYs) { trainYs.dispose(); trainYs = null; }
        if (testXs) { testXs.dispose(); testXs = null; }
        if (testYs) { testYs.dispose(); testYs = null; }
    }

    // clear UI and model
    async function resetAll() {
        disposeDataTensors();
        if (currentModel) {
            currentModel.dispose();
            currentModel = null;
        }
        // reset preview placeholders
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

    // ---- load data handler ----
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

            // dispose old tensors before replacing
            disposeDataTensors();
            trainXs = train.xs; trainYs = train.ys;
            testXs = test.xs; testYs = test.ys;

            dataStatus.innerText = `📊 train: ${trainXs.shape[0]}  |  test: ${testXs.shape[0]}`;
            log(`✅ train samples: ${trainXs.shape[0]}, test: ${testXs.shape[0]}`);
            // initialize default model if none
            if (!currentModel) {
                currentModel = createModel();
                logModelSummary();
            }
        } catch (e) {
            log(`❌ load error: ${e.message}`);
            console.error(e);
        }
    });

    // ---- build CNN model ----
    function createModel() {
        const model = tf.sequential();
        model.add(tf.layers.conv2d({
            filters: 32, kernelSize: 3, activation: 'relu', padding: 'same',
            inputShape: [28, 28, 1]
        }));
        model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu', padding: 'same' }));
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
        return model;
    }

    function logModelSummary() {
        if (!currentModel) return;
        const summary = [];
        currentModel.layers.forEach((l, i) => {
            summary.push(`${i}: ${l.name} ${l.outputShape}`);
        });
        modelSummaryDiv.innerText = summary.join('\n') || 'model built';
    }

    // ---- training with tfvis callbacks ----
    trainBtn.addEventListener('click', async () => {
        if (!trainXs || !trainYs) { alert('load data first'); return; }
        if (!currentModel) currentModel = createModel();

        try {
            const { trainXs: subTrainXs, trainYs: subTrainYs, valXs, valYs } =
                window.dataLoader.splitTrainVal(trainXs, trainYs, 0.1);

            log(`🏋️ training on ${subTrainXs.shape[0]} samples, val ${valXs.shape[0]}`);

            // tfvis training callbacks (loss & accuracy curves)
            const container = { name: 'MNIST Training', tab: 'Training' };
            const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
            const callbacks = tfvis.show.fitCallbacks(container, metrics, {
                callbacks: ['onEpochEnd'],
                width: 400, height: 300
            });

            const start = performance.now();
            await currentModel.fit(subTrainXs, subTrainYs, {
                batchSize: 128,
                epochs: 6,
                validationData: [valXs, valYs],
                shuffle: true,
                callbacks: callbacks
            });
            const duration = ((performance.now() - start) / 1000).toFixed(1);
            log(`✅ training finished in ${duration}s`);

            // dispose split tensors
            tf.dispose([subTrainXs, subTrainYs, valXs, valYs]);
        } catch (err) {
            log(`❌ train error: ${err.message}`);
        }
    });

    // ---- evaluation (confusion matrix + per‑class acc) ----
    evaluateBtn.addEventListener('click', async () => {
        if (!testXs || !testYs || !currentModel) {
            alert('need model and test data');
            return;
        }
        try {
            log('🔍 evaluating on test set...');
            const evalResult = await currentModel.evaluate(testXs, testYs, { batchSize: 128 });
            const acc = evalResult[1].dataSync()[0];
            accStatus.innerText = `accuracy: ${(acc * 100).toFixed(2)}%`;
            log(`test accuracy = ${(acc * 100).toFixed(2)}%`);

            // ---- confusion matrix & per‑class accuracy (tfvis) ----
            const preds = currentModel.predict(testXs);
            const trueLabels = testYs.argMax(-1);
            const predLabels = preds.argMax(-1);

            // per‑class accuracy: we need to mask correctly
            const equality = trueLabels.equal(predLabels).cast('float32'); // 1 correct, 0 wrong

            // one‑hot true labels to compute per‑class
            const trueOneHot = tf.oneHot(trueLabels, 10); // [N,10]
            const classMask = trueOneHot.mul(equality.expandDims(1)); // correct per class
            const classCorrect = classMask.sum(0); // sum correct per class
            const classCounts = trueOneHot.sum(0);
            const classAcc = classCorrect.div(classCounts); // [10] tensor

            const classAccArray = await classAcc.array();

            // render bar chart with tfvis
            tfvis.render.barchart(
                { name: 'Per‑class accuracy', tab: 'Evaluation' },
                classAccArray.map((v, i) => ({ index: i, accuracy: v })),
                { xLabel: 'class', yLabel: 'accuracy' }
            );

            // confusion matrix
            const confusionMatrix = await tf.math.confusionMatrix(trueLabels, predLabels, 10);
            tfvis.render.confusionMatrix(
                { name: 'Confusion matrix', tab: 'Evaluation' },
                confusionMatrix,
                { shadeDiagonal: true, width: 400 }
            );

            // dispose intermediates
            tf.dispose([preds, trueLabels, predLabels, equality, trueOneHot, classMask, classCorrect, classCounts, classAcc, confusionMatrix]);

        } catch (err) {
            log(`❌ eval error: ${err.message}`);
        }
    });

    // ---- test 5 random preview ----
    test5Btn.addEventListener('click', async () => {
        if (!testXs || !testYs || !currentModel) {
            alert('load test data and model first');
            return;
        }
        try {
            const { batchXs, batchYs } = window.dataLoader.getRandomTestBatch(testXs, testYs, 5);
            const predProbs = currentModel.predict(batchXs);
            const predLabels = predProbs.argMax(-1); // [5] int
            const trueLabels = batchYs.argMax(-1);

            const predArr = await predLabels.array();
            const trueArr = await trueLabels.array();

            // clear preview container and rebuild 5 items
            previewContainer.innerHTML = '';
            for (let i = 0; i < 5; i++) {
                const div = document.createElement('div');
                div.className = 'preview-item';

                const canvas = document.createElement('canvas');
                canvas.width = 112; canvas.height = 112;
                canvas.style.width = '84px'; canvas.style.height = '84px';

                // extract single image tensor
                const imgTensor = batchXs.slice([i,0,0,0], [1,28,28,1]).squeeze();
                window.dataLoader.draw28x28ToCanvas(imgTensor, canvas, 4);
                imgTensor.dispose();

                const labelSpan = document.createElement('div');
                labelSpan.className = `pred-label ${predArr[i] === trueArr[i] ? 'correct' : 'wrong'}`;
                labelSpan.innerText = `p:${predArr[i]} (true:${trueArr[i]})`;

                div.appendChild(canvas);
                div.appendChild(labelSpan);
                previewContainer.appendChild(div);
            }
            tf.dispose([batchXs, batchYs, predProbs, predLabels, trueLabels]);
        } catch (err) {
            log(`❌ preview error: ${err.message}`);
        }
    });

    // ---- save model (download) ----
    saveModelBtn.addEventListener('click', async () => {
        if (!currentModel) { alert('no model to save'); return; }
        try {
            await currentModel.save('downloads://mnist-cnn');
            log('💾 model saved (downloads://mnist-cnn)');
        } catch (e) { log('save error: ' + e); }
    });

    // ---- load model from user files (.json + .bin) ----
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
            currentModel.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
            logModelSummary();
            log('✅ model loaded from files');
        } catch (e) {
            log(`❌ load model error: ${e.message}`);
        }
        modelJsonInput.value = ''; // allow reload same files
    });

    // ---- reset ----
    resetBtn.addEventListener('click', resetAll);

    // ---- toggle tfvis visor ----
    toggleVisorBtn.addEventListener('click', () => tfvis.visor().toggle());

    // initial summary if model created later
})();