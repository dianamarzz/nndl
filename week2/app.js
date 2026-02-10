// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let featureImportances = null;

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// Improved CSV parsing with proper comma handling
function parseCSV(csvText) {
    const lines = [];
    let currentLine = [];
    let inQuotes = false;
    let currentValue = '';
    
    for (let i = 0; i < csvText.length; i++) {
        const char = csvText[i];
        const nextChar = csvText[i + 1];
        
        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote inside quoted field
                currentValue += '"';
                i++; // Skip next character
            } else {
                // Start or end of quoted field
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // End of field
            currentLine.push(currentValue.trim());
            currentValue = '';
        } else if (char === '\n' && !inQuotes) {
            // End of line
            currentLine.push(currentValue.trim());
            lines.push(currentLine);
            currentLine = [];
            currentValue = '';
        } else if (char === '\r' && nextChar === '\n' && !inQuotes) {
            // Handle Windows line endings
            currentLine.push(currentValue.trim());
            lines.push(currentLine);
            currentLine = [];
            currentValue = '';
            i++; // Skip the \n
        } else {
            currentValue += char;
        }
    }
    
    // Add last line if exists
    if (currentValue !== '' || currentLine.length > 0) {
        currentLine.push(currentValue.trim());
        lines.push(currentLine);
    }
    
    // Extract headers and data
    const headers = lines[0];
    const data = lines.slice(1).map(line => {
        const obj = {};
        headers.forEach((header, index) => {
            let value = line[index] || null;
            
            // Convert to number if possible
            if (value !== null && !isNaN(value) && value.trim() !== '') {
                value = parseFloat(value);
            } else if (value === '' || value === 'NULL' || value === 'null') {
                value = null;
            }
            
            obj[header.trim()] = value;
        });
        return obj;
    });
    
    return data;
}

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        updateStatus('data-status', 'Please upload both training and test CSV files.', 'warning');
        return;
    }
    
    updateStatus('data-status', 'Loading and parsing data...', 'info');
    
    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);
        
        updateStatus('data-status', 
            `Data loaded successfully!<br>Training: ${trainData.length} samples<br>Test: ${testData.length} samples`, 
            'success');
        
        document.getElementById('inspect-btn').disabled = false;
        document.getElementById('inspect-btn').classList.remove('disabled');
    } catch (error) {
        updateStatus('data-status', `Error loading data: ${error.message}`, 'error');
        console.error('Load error:', error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Update status with styling
function updateStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    const colors = {
        info: '#1a73e8',
        success: '#0d8a3c',
        warning: '#f09300',
        error: '#d32f2f'
    };
    
    element.innerHTML = `<div style="color: ${colors[type]}; padding: 10px; border-left: 4px solid ${colors[type]}; background: ${colors[type]}10;">
        ${message}
    </div>`;
}

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        updateStatus('data-status', 'Please load data first.', 'warning');
        return;
    }
    
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Show basic statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <h4>Dataset Shape</h4>
                <p>${trainData.length} rows × ${Object.keys(trainData[0]).length} columns</p>
            </div>
            <div class="stat-card">
                <h4>Survival Rate</h4>
                <p>${calculateSurvivalRate()}% survived</p>
            </div>
            <div class="stat-card">
                <h4>Missing Values</h4>
                <p>${calculateMissingPercentage()}% overall</p>
            </div>
        </div>
    `;
    
    // Create visualizations
    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
    document.getElementById('preprocess-btn').classList.remove('disabled');
}

// Calculate survival rate
function calculateSurvivalRate() {
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    return ((survivalCount / trainData.length) * 100).toFixed(1);
}

// Calculate missing percentage
function calculateMissingPercentage() {
    let totalMissing = 0;
    let totalValues = 0;
    
    Object.keys(trainData[0]).forEach(feature => {
        trainData.forEach(row => {
            totalValues++;
            if (row[feature] === null || row[feature] === undefined) {
                totalMissing++;
            }
        });
    });
    
    return ((totalMissing / totalValues) * 100).toFixed(1);
}

// Create preview table
function createPreviewTable(data) {
    const table = document.createElement('table');
    table.className = 'data-table';
    
    const headers = Object.keys(data[0]);
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    data.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            const value = row[header];
            td.textContent = value !== null ? value : '—';
            td.title = value !== null ? value : 'Missing value';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Key Insights</h3>';
    
    // Calculate key survival factors
    const survivalBySex = calculateSurvivalByFeature('Sex');
    const survivalByClass = calculateSurvivalByFeature('Pclass');
    const survivalByAgeGroup = calculateSurvivalByAgeGroup();
    
    // Create visualization cards
    const insightsHTML = `
        <div class="insights-grid">
            <div class="insight-card">
                <h4>Gender Impact</h4>
                <div class="insight-bar">
                    <div class="bar-label">Female: ${survivalBySex.female}%</div>
                    <div class="bar" style="width: ${survivalBySex.female}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">Male: ${survivalBySex.male}%</div>
                    <div class="bar" style="width: ${survivalBySex.male}%"></div>
                </div>
            </div>
            
            <div class="insight-card">
                <h4>Class Impact</h4>
                <div class="insight-bar">
                    <div class="bar-label">1st Class: ${survivalByClass[1]}%</div>
                    <div class="bar" style="width: ${survivalByClass[1]}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">2nd Class: ${survivalByClass[2]}%</div>
                    <div class="bar" style="width: ${survivalByClass[2]}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">3rd Class: ${survivalByClass[3]}%</div>
                    <div class="bar" style="width: ${survivalByClass[3]}%"></div>
                </div>
            </div>
            
            <div class="insight-card">
                <h4>Age Group Impact</h4>
                <div class="insight-bar">
                    <div class="bar-label">Children: ${survivalByAgeGroup.child}%</div>
                    <div class="bar" style="width: ${survivalByAgeGroup.child}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">Adults: ${survivalByAgeGroup.adult}%</div>
                    <div class="bar" style="width: ${survivalByAgeGroup.adult}%"></div>
                </div>
            </div>
        </div>
    `;
    
    chartsDiv.innerHTML += insightsHTML;
}

// Helper functions for insights
function calculateSurvivalByFeature(feature) {
    const groups = {};
    trainData.forEach(row => {
        if (row[feature] && row.Survived !== undefined) {
            const value = row[feature];
            if (!groups[value]) {
                groups[value] = { survived: 0, total: 0 };
            }
            groups[value].total++;
            if (row.Survived === 1) {
                groups[value].survived++;
            }
        }
    });
    
    const result = {};
    Object.entries(groups).forEach(([key, stats]) => {
        result[key] = ((stats.survived / stats.total) * 100).toFixed(1);
    });
    
    return result;
}

function calculateSurvivalByAgeGroup() {
    const groups = { child: { survived: 0, total: 0 }, adult: { survived: 0, total: 0 } };
    
    trainData.forEach(row => {
        if (row.Age !== null && row.Survived !== undefined) {
            const group = row.Age < 18 ? 'child' : 'adult';
            groups[group].total++;
            if (row.Survived === 1) {
                groups[group].survived++;
            }
        }
    });
    
    return {
        child: groups.child.total > 0 ? ((groups.child.survived / groups.child.total) * 100).toFixed(1) : '0.0',
        adult: groups.adult.total > 0 ? ((groups.adult.survived / groups.adult.total) * 100).toFixed(1) : '0.0'
    };
}

// Preprocess data
function preprocessData() {
    if (!trainData || !testData) {
        updateStatus('preprocessing-output', 'Please load data first.', 'warning');
        return;
    }
    
    updateStatus('preprocessing-output', 'Preprocessing data...', 'info');
    
    try {
        // Calculate imputation values
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(a => a !== null));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(f => f !== null));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
        // Preprocess training data
        preprocessedTrainData = { features: [], labels: [] };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = { features: [], passengerIds: [] };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        updateStatus('preprocessing-output', `
            <div class="success-message">
                <h4>✓ Preprocessing Complete</h4>
                <p>Training features: ${preprocessedTrainData.features.shape[0]} × ${preprocessedTrainData.features.shape[1]}</p>
                <p>Training labels: ${preprocessedTrainData.labels.shape[0]}</p>
                <p>Test samples: ${preprocessedTestData.features.length}</p>
            </div>
        `, 'success');
        
        document.getElementById('create-model-btn').disabled = false;
        document.getElementById('create-model-btn').classList.remove('disabled');
    } catch (error) {
        updateStatus('preprocessing-output', `Error during preprocessing: ${error.message}`, 'error');
        console.error('Preprocessing error:', error);
    }
}

// Extract features with sigmoid gating for importance
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    
    // Standardize numerical features
    const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null)) || 1;
    const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null)) || 1;
    
    const standardizedAge = (age - ageMedian) / ageStd;
    const standardizedFare = (fare - fareMedian) / fareStd;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
    // Add family features if enabled
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    
    return features;
}

// Statistics helper functions
function calculateMedian(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function calculateMode(values) {
    if (values.length === 0) return null;
    const freq = {};
    let max = 0;
    let mode = null;
    
    values.forEach(value => {
        freq[value] = (freq[value] || 0) + 1;
        if (freq[value] > max) {
            max = freq[value];
            mode = value;
        }
    });
    
    return mode;
}

function calculateStdDev(values) {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(variance);
}

function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) encoding[index] = 1;
    return encoding;
}

// Create model with sigmoid gating for feature importance
function createModel() {
    if (!preprocessedTrainData) {
        updateStatus('model-summary', 'Please preprocess data first.', 'warning');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create model with sigmoid gating layer
    model = tf.sequential();
    
    // Add sigmoid gating layer for feature importance
    model.add(tf.layers.dense({
        units: inputShape,
        activation: 'sigmoid',
        inputShape: [inputShape],
        kernelInitializer: 'ones',
        biasInitializer: 'zeros',
        name: 'feature_gate'
    }));
    
    // Main model layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = `
        <div class="model-info">
            <h3>Model Architecture</h3>
            <div class="model-layers">
                <div class="layer">
                    <span class="layer-name">Sigmoid Gating Layer</span>
                    <span class="layer-details">${inputShape} units</span>
                </div>
                <div class="layer">
                    <span class="layer-name">Hidden Layer</span>
                    <span class="layer-details">16 units, ReLU activation</span>
                </div>
                <div class="layer">
                    <span class="layer-name">Output Layer</span>
                    <span class="layer-details">1 unit, Sigmoid activation</span>
                </div>
            </div>
            <p class="param-count">Total parameters: ${model.countParams().toLocaleString()}</p>
        </div>
    `;
    
    document.getElementById('train-btn').disabled = false;
    document.getElementById('train-btn').classList.remove('disabled');
}

// Train model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        updateStatus('training-status', 'Please create model first.', 'warning');
        return;
    }
    
    updateStatus('training-status', 'Training model...', 'info');
    
    try {
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        validationData = valFeatures;
        validationLabels = valLabels;
        
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = document.getElementById('training-progress');
                    if (progress) {
                        progress.innerHTML = `
                            Epoch ${epoch + 1}/50
                            <br>Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(1)}%
                            <br>Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(1)}%
                        `;
                    }
                },
                onTrainEnd: () => {
                    updateStatus('training-status', `
                        <div class="success-message">
                            <h4>✓ Training Complete</h4>
                            <p>Model trained successfully on ${trainFeatures.shape[0]} samples</p>
                        </div>
                    `, 'success');
                    
                    // Extract feature importances
                    extractFeatureImportances();
                    
                    // Make predictions
                    validationPredictions = model.predict(validationData);
                    
                    // Enable evaluation
                    document.getElementById('threshold-slider').disabled = false;
                    document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
                    
                    // Calculate initial metrics
                    updateMetrics();
                    
                    document.getElementById('predict-btn').disabled = false;
                    document.getElementById('predict-btn').classList.remove('disabled');
                }
            }
        });
        
        // Add progress indicator
        const statusDiv = document.getElementById('training-status');
        statusDiv.innerHTML += '<div id="training-progress" class="progress-info"></div>';
        
    } catch (error) {
        updateStatus('training-status', `Error during training: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

// Extract feature importances from sigmoid gate layer
function extractFeatureImportances() {
    try {
        const gateLayer = model.getLayer('feature_gate');
        const weights = gateLayer.getWeights()[0]; // Get kernel weights
        const importances = weights.arraySync();
        
        // Calculate average importance for each feature
        featureImportances = [];
        for (let i = 0; i < importances.length; i++) {
            let sum = 0;
            for (let j = 0; j < importances[i].length; j++) {
                sum += Math.abs(importances[i][j]);
            }
            featureImportances.push(sum / importances[i].length);
        }
        
        // Display top features
        displayTopFeatures();
    } catch (error) {
        console.error('Error extracting feature importances:', error);
    }
}

// Display top features
function displayTopFeatures() {
    const features = [
        'Age', 'Fare', 'SibSp', 'Parch',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_male', 'Sex_female',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'FamilySize', 'IsAlone'
    ].slice(0, featureImportances.length);
    
    // Combine importances with feature names
    const featureData = features.map((name, idx) => ({
        name,
        importance: featureImportances[idx] || 0
    }));
    
    // Sort by importance
    featureData.sort((a, b) => b.importance - a.importance);
    
    // Display in insights section
    const insightsDiv = document.getElementById('model-insights');
    insightsDiv.innerHTML = `
        <h3>Feature Importance Analysis</h3>
        <p>Based on sigmoid gate layer activations:</p>
        <div class="importance-chart">
            ${featureData.slice(0, 5).map(f => `
                <div class="importance-item">
                    <div class="importance-label">${f.name}</div>
                    <div class="importance-bar-container">
                        <div class="importance-bar" style="width: ${(f.importance * 100).toFixed(1)}%"></div>
                        <span class="importance-value">${(f.importance * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `).join('')}
        </div>
        <div class="key-finding">
            <h4>Key Finding:</h4>
            <p><strong>${featureData[0].name}</strong> is the most important predictor of survival, 
            with ${(featureData[0].importance * 100).toFixed(1)}% relative importance.</p>
            ${featureData[0].name.includes('Sex') ? 
                '<p>This aligns with historical records showing "women and children first" evacuation protocol.</p>' : ''}
            ${featureData[0].name.includes('Pclass') ? 
                '<p>This reflects the advantage of higher-class passengers in accessing lifeboats.</p>' : ''}
        </div>
    `;
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    const predVals = await validationPredictions.array();
    const trueVals = await validationLabels.array();
    
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else fn++;
    }
    
    // Update confusion matrix
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <div class="confusion-matrix">
            <div class="cm-header">
                <div></div>
                <div class="cm-predicted">Predicted Positive</div>
                <div class="cm-predicted">Predicted Negative</div>
            </div>
            <div class="cm-row">
                <div class="cm-actual">Actual Positive</div>
                <div class="cm-cell tp">${tp}<br><small>True Positive</small></div>
                <div class="cm-cell fn">${fn}<br><small>False Negative</small></div>
            </div>
            <div class="cm-row">
                <div class="cm-actual">Actual Negative</div>
                <div class="cm-cell fp">${fp}<br><small>False Positive</small></div>
                <div class="cm-cell tn">${tn}<br><small>True Negative</small></div>
            </div>
        </div>
    `;
    
    // Calculate metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <div class="metrics-grid">
            <div class="metric">
                <h4>Accuracy</h4>
                <p class="metric-value">${(accuracy * 100).toFixed(2)}%</p>
            </div>
            <div class="metric">
                <h4>Precision</h4>
                <p class="metric-value">${precision.toFixed(4)}</p>
            </div>
            <div class="metric">
                <h4>Recall</h4>
                <p class="metric-value">${recall.toFixed(4)}</p>
            </div>
            <div class="metric">
                <h4>F1 Score</h4>
                <p class="metric-value">${f1.toFixed(4)}</p>
            </div>
        </div>
    `;
}

// Predict on test data
async function predict() {
    if (!model || !preprocessedTestData) {
        updateStatus('prediction-output', 'Please train model first.', 'warning');
        return;
    }
    
    updateStatus('prediction-output', 'Making predictions...', 'info');
    
    try {
        const testFeatures = tf.tensor2d(preprocessedTestData.features);
        testPredictions = model.predict(testFeatures);
        const predValues = await testPredictions.array();
        
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i]
        }));
        
        // Display predictions
        const outputDiv = document.getElementById('prediction-output');
        outputDiv.innerHTML = `
            <div class="success-message">
                <h4>✓ Predictions Generated</h4>
                <p>Successfully predicted ${results.length} test samples</p>
            </div>
            <h4>Sample Predictions</h4>
            ${createPredictionTable(results.slice(0, 5))}
        `;
        
        // Calculate prediction statistics
        const survivalPredictions = results.filter(r => r.Survived === 1).length;
        outputDiv.innerHTML += `
            <div class="prediction-stats">
                <p><strong>Predicted Survival Rate:</strong> ${((survivalPredictions / results.length) * 100).toFixed(1)}%</p>
                <p><strong>Average Confidence:</strong> ${(results.reduce((sum, r) => sum + r.Probability, 0) / results.length * 100).toFixed(1)}%</p>
            </div>
        `;
        
        document.getElementById('export-btn').disabled = false;
        document.getElementById('export-btn').classList.remove('disabled');
        
    } catch (error) {
        updateStatus('prediction-output', `Error during prediction: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

// Create prediction table HTML
function createPredictionTable(data) {
    let html = '<table class="prediction-table"><tr><th>Passenger ID</th><th>Survived</th><th>Probability</th><th>Confidence</th></tr>';
    
    data.forEach(row => {
        const survivedText = row.Survived === 1 ? '✓ Yes' : '✗ No';
        const confidence = row.Probability >= 0.7 ? 'High' : row.Probability >= 0.4 ? 'Medium' : 'Low';
        const confidenceClass = confidence.toLowerCase();
        
        html += `
            <tr>
                <td>${row.PassengerId}</td>
                <td class="${row.Survived === 1 ? 'survived-yes' : 'survived-no'}">${survivedText}</td>
                <td>${row.Probability.toFixed(4)}</td>
                <td class="confidence-${confidenceClass}">${confidence}</td>
            </tr>
        `;
    });
    
    html += '</table>';
    return html;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        updateStatus('export-status', 'Please make predictions first.', 'warning');
        return;
    }
    
    updateStatus('export-status', 'Preparing files for download...', 'info');
    
    try {
        const predValues = await testPredictions.array();
        
        // Create submission CSV
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i] >= 0.5 ? 1 : 0}\n`;
        });
        
        // Create download link
        const blob = new Blob([submissionCSV], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'titanic_predictions.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        updateStatus('export-status', `
            <div class="success-message">
                <h4>✓ Export Complete</h4>
                <p>File "titanic_predictions.csv" has been downloaded</p>
                <p>Format: PassengerId,Survived (0 = Did not survive, 1 = Survived)</p>
            </div>
        `, 'success');
        
    } catch (error) {
        updateStatus('export-status', `Error during export: ${error.message}`, 'error');
    }
}
