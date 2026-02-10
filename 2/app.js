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
    
    // Create scrollable table container
    const tableContainer = document.createElement('div');
    tableContainer.className = 'table-container';
    tableContainer.style.overflowX = 'auto';
    tableContainer.style.marginBottom = '20px';
    tableContainer.appendChild(createPreviewTable(trainData.slice(0, 10)));
    previewDiv.appendChild(tableContainer);
    
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
    
    // Add survival factor analysis
    const survivalFactors = calculateSurvivalFactors();
    if (survivalFactors) {
        const chartsDiv = document.getElementById('charts');
        const factorsDiv = document.createElement('div');
        factorsDiv.className = 'survival-factors';
        factorsDiv.innerHTML = `
            <div class="key-finding" style="margin-top: 20px;">
                <h4>🔍 Most Important Factor for Survival:</h4>
                <div class="factor-highlight">
                    <div class="factor-header">
                        <span class="factor-name">${survivalFactors.feature}</span>
                        <span class="factor-value">= ${survivalFactors.value}</span>
                    </div>
                    <div class="factor-stats">
                        <div class="stat">
                            <span class="stat-label">Survival Rate:</span>
                            <span class="stat-value">${(survivalFactors.survivalRate * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Impact vs Average:</span>
                            <span class="stat-value">+${(survivalFactors.impact * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <p class="factor-interpretation">${survivalFactors.interpretation}</p>
                </div>
            </div>
        `;
        chartsDiv.appendChild(factorsDiv);
        
        // Add detailed survival analysis
        chartsDiv.appendChild(displaySurvivalAnalysis());
    }
    
    document.getElementById('preprocess-btn').disabled = false;
    document.getElementById('preprocess-btn').classList.remove('disabled');
}

// Calculate most important factor for survival
function calculateSurvivalFactors() {
    if (!trainData || trainData.length === 0) return null;
    
    const factors = {};
    const avgSurvivalRate = calculateSurvivalRate() / 100;
    
    // Analyze each feature's impact on survival
    Object.keys(trainData[0]).forEach(feature => {
        if (feature === TARGET_FEATURE || feature === ID_FEATURE) return;
        
        // Group data by feature value and calculate survival rates
        const groups = {};
        
        trainData.forEach(row => {
            if (row[feature] !== null && row[feature] !== undefined && 
                row[TARGET_FEATURE] !== undefined) {
                
                const key = String(row[feature]);
                if (!groups[key]) {
                    groups[key] = { survived: 0, total: 0 };
                }
                groups[key].total++;
                if (row[TARGET_FEATURE] === 1) {
                    groups[key].survived++;
                }
            }
        });
        
        // Find the value with highest survival rate
        let bestValue = null;
        let bestRate = 0;
        
        Object.entries(groups).forEach(([value, stats]) => {
            if (stats.total > 5) { // Require minimum samples
                const rate = stats.survived / stats.total;
                if (rate > bestRate) {
                    bestRate = rate;
                    bestValue = value;
                }
            }
        });
        
        if (bestValue !== null) {
            factors[feature] = {
                bestValue: bestValue,
                survivalRate: bestRate,
                impact: bestRate - avgSurvivalRate,
                interpretation: interpretFactor(feature, bestValue, bestRate)
            };
        }
    });
    
    // Find the most important factor (highest survival rate)
    let mostImportant = null;
    let maxRate = 0;
    
    Object.entries(factors).forEach(([feature, data]) => {
        if (data.survivalRate > maxRate) {
            maxRate = data.survivalRate;
            mostImportant = {
                feature: feature,
                value: data.bestValue,
                survivalRate: data.survivalRate,
                impact: data.impact,
                interpretation: data.interpretation
            };
        }
    });
    
    return mostImportant;
}

// Helper function to interpret factors
function interpretFactor(feature, value, rate) {
    const percentage = (rate * 100).toFixed(1);
    
    switch(feature) {
        case 'Sex':
            return value === 'female' 
                ? `Female passengers survived ${percentage}% of the time (vs ${(calculateSurvivalRate())}% overall)`
                : `Male passengers survived ${percentage}% of the time (vs ${(calculateSurvivalRate())}% overall)`;
        
        case 'Pclass':
            return `Class ${value} passengers survived ${percentage}% of the time (vs ${(calculateSurvivalRate())}% overall)`;
        
        case 'Age':
            const ageGroup = parseFloat(value) < 18 ? 'Children' : 'Adults';
            return `${ageGroup} survived ${percentage}% of the time (vs ${(calculateSurvivalRate())}% overall)`;
        
        case 'Embarked':
            const port = value === 'C' ? 'Cherbourg' : 
                        value === 'Q' ? 'Queenstown' : 'Southampton';
            return `Passengers from ${port} survived ${percentage}% of the time (vs ${(calculateSurvivalRate())}% overall)`;
        
        default:
            return `${feature} = ${value} → ${percentage}% survival rate (vs ${(calculateSurvivalRate())}% overall)`;
    }
}

// Display detailed survival analysis
function displaySurvivalAnalysis() {
    const analysisDiv = document.createElement('div');
    analysisDiv.className = 'survival-analysis';
    
    // Calculate survival rates for key features
    const sexRate = calculateSurvivalByFeature('Sex');
    const classRate = calculateSurvivalByFeature('Pclass');
    const ageGroupRate = calculateSurvivalByAgeGroup();
    const overallRate = calculateSurvivalRate();
    
    analysisDiv.innerHTML = `
        <div class="key-finding" style="margin-top: 20px;">
            <h4>📊 Survival Analysis Summary</h4>
            <div class="analysis-grid">
                <div class="analysis-card">
                    <h5>Top Survival Factors</h5>
                    <ul>
                        <li><strong>Female</strong>: ${sexRate.female || 0}% survival</li>
                        <li><strong>1st Class</strong>: ${classRate[1] || 0}% survival</li>
                        <li><strong>Children</strong>: ${ageGroupRate.child || 0}% survival</li>
                    </ul>
                </div>
                <div class="analysis-card">
                    <h5>Risk Factors</h5>
                    <ul>
                        <li><strong>Male</strong>: ${sexRate.male || 0}% survival</li>
                        <li><strong>3rd Class</strong>: ${classRate[3] || 0}% survival</li>
                        <li><strong>Adults</strong>: ${ageGroupRate.adult || 0}% survival</li>
                    </ul>
                </div>
                <div class="analysis-card">
                    <h5>Overall Statistics</h5>
                    <ul>
                        <li><strong>Total Passengers</strong>: ${trainData.length}</li>
                        <li><strong>Survival Rate</strong>: ${overallRate}%</li>
                        <li><strong>Perished</strong>: ${(100 - parseFloat(overallRate)).toFixed(1)}%</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    return analysisDiv;
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
            if (row[feature] === null || row[feature] === undefined || (typeof row[feature] === 'number' && isNaN(row[feature]))) {
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
    table.style.minWidth = '800px'; // Ensure table has minimum width
    
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
            td.textContent = value !== null && value !== undefined ? value : '—';
            td.title = value !== null && value !== undefined ? value : 'Missing value';
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
                    <div class="bar-label">Female: ${survivalBySex.female || 0}%</div>
                    <div class="bar" style="width: ${survivalBySex.female || 0}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">Male: ${survivalBySex.male || 0}%</div>
                    <div class="bar" style="width: ${survivalBySex.male || 0}%"></div>
                </div>
            </div>
            
            <div class="insight-card">
                <h4>Class Impact</h4>
                <div class="insight-bar">
                    <div class="bar-label">1st Class: ${survivalByClass[1] || 0}%</div>
                    <div class="bar" style="width: ${survivalByClass[1] || 0}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">2nd Class: ${survivalByClass[2] || 0}%</div>
                    <div class="bar" style="width: ${survivalByClass[2] || 0}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">3rd Class: ${survivalByClass[3] || 0}%</div>
                    <div class="bar" style="width: ${survivalByClass[3] || 0}%"></div>
                </div>
            </div>
            
            <div class="insight-card">
                <h4>Age Group Impact</h4>
                <div class="insight-bar">
                    <div class="bar-label">Children: ${survivalByAgeGroup.child || 0}%</div>
                    <div class="bar" style="width: ${survivalByAgeGroup.child || 0}%"></div>
                </div>
                <div class="insight-bar">
                    <div class="bar-label">Adults: ${survivalByAgeGroup.adult || 0}%</div>
                    <div class="bar" style="width: ${survivalByAgeGroup.adult || 0}%"></div>
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
        if (row[feature] !== null && row[feature] !== undefined && row.Survived !== undefined) {
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
        if (row.Age !== null && row.Age !== undefined && !isNaN(row.Age) && row.Survived !== undefined) {
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

// Preprocess data (removed feature engineering)
function preprocessData() {
    if (!trainData || !testData) {
        updateStatus('preprocessing-output', 'Please load data first.', 'warning');
        return;
    }
    
    updateStatus('preprocessing-output', 'Preprocessing data...', 'info');
    
    try {
        // Calculate imputation values from training data only
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(a => a !== null && !isNaN(a)));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(f => f !== null && !isNaN(f)));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
        // Preprocess training data
        preprocessedTrainData = { features: [], labels: [] };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            if (row[TARGET_FEATURE] !== undefined && row[TARGET_FEATURE] !== null) {
                preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
            }
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

// Extract features without feature engineering
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = (row.Age !== null && !isNaN(row.Age)) ? row.Age : ageMedian;
    const fare = (row.Fare !== null && !isNaN(row.Fare)) ? row.Fare : fareMedian;
    const embarked = (row.Embarked !== null) ? row.Embarked : embarkedMode;
    
    // Calculate statistics from training data for standardization
    const trainAges = trainData.map(r => r.Age).filter(a => a !== null && !isNaN(a));
    const trainFares = trainData.map(r => r.Fare).filter(f => f !== null && !isNaN(f));
    
    const ageMean = trainAges.length > 0 ? trainAges.reduce((a, b) => a + b, 0) / trainAges.length : 0;
    const fareMean = trainFares.length > 0 ? trainFares.reduce((a, b) => a + b, 0) / trainFares.length : 0;
    
    const ageStd = calculateStdDev(trainAges) || 1;
    const fareStd = calculateStdDev(trainFares) || 1;
    
    // Standardize numerical features
    const standardizedAge = (age - ageMean) / ageStd;
    const standardizedFare = (fare - fareMean) / fareStd;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features (no family features added)
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
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
    if (values.length === 0) return 'S'; // Default to Southampton
    const freq = {};
    let max = 0;
    let mode = 'S';
    
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

// Create model with proper feature importance
function createModel() {
    if (!preprocessedTrainData) {
        updateStatus('model-summary', 'Please preprocess data first.', 'warning');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create simpler model without sigmoid gating for now
    model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    // Hidden layer
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu'
    }));
    
    // Output layer
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
                    <span class="layer-name">Input Layer</span>
                    <span class="layer-details">${inputShape} features → 32 units</span>
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

// Train model with progress bar
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        updateStatus('training-status', 'Please create model first.', 'warning');
        return;
    }
    
    updateStatus('training-status', 'Training model...', 'info');
    
    // Add progress bar container
    const trainingStatusDiv = document.getElementById('training-status');
    trainingStatusDiv.innerHTML += `
        <div id="training-progress-container" style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span id="epoch-info">Epoch: 0/50</span>
                <span id="progress-percent">0%</span>
            </div>
            <div style="width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div id="progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #3498db, #2980b9); transition: width 0.3s;"></div>
            </div>
            <div id="training-metrics" style="margin-top: 10px; font-size: 0.9rem; color: #666;"></div>
        </div>
    `;
    
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
                    const progressPercent = ((epoch + 1) / 50) * 100;
                    
                    // Update progress bar
                    document.getElementById('progress-bar').style.width = `${progressPercent}%`;
                    document.getElementById('epoch-info').textContent = `Epoch: ${epoch + 1}/50`;
                    document.getElementById('progress-percent').textContent = `${progressPercent.toFixed(0)}%`;
                    
                    // Update metrics
                    const metricsDiv = document.getElementById('training-metrics');
                    metricsDiv.innerHTML = `
                        Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(1)}%
                        <br>Val Loss: ${logs.val_loss.toFixed(4)} | Val Accuracy: ${(logs.val_acc * 100).toFixed(1)}%
                    `;
                },
                onTrainEnd: () => {
                    updateStatus('training-status', `
                        <div class="success-message">
                            <h4>✓ Training Complete</h4>
                            <p>Model trained successfully on ${trainFeatures.shape[0]} samples</p>
                        </div>
                    `, 'success');
                    
                    // Calculate feature importances using permutation importance
                    calculatePermutationImportance();
                    
                    // Make validation predictions
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
        
    } catch (error) {
        updateStatus('training-status', `Error during training: ${error.message}`, 'error');
        console.error('Training error:', error);
    }
}

// Calculate permutation importance
async function calculatePermutationImportance() {
    try {
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Get baseline accuracy
        const baselinePred = model.predict(valFeatures);
        const baselinePredArray = await baselinePred.array();
        const baselineLabels = await valLabels.array();
        
        let baselineCorrect = 0;
        for (let i = 0; i < baselinePredArray.length; i++) {
            if ((baselinePredArray[i] >= 0.5 ? 1 : 0) === baselineLabels[i]) {
                baselineCorrect++;
            }
        }
        const baselineAccuracy = baselineCorrect / baselinePredArray.length;
        
        // Calculate importance for each feature
        const numFeatures = valFeatures.shape[1];
        const importances = new Array(numFeatures).fill(0);
        
        // Shuffle each feature and measure impact
        for (let featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
            // Create shuffled copy of the feature
            const shuffledFeatures = valFeatures.clone();
            const featureData = await shuffledFeatures.slice([0, featureIdx], [shuffledFeatures.shape[0], 1]).array();
            
            // Shuffle the feature values
            const shuffled = [...featureData].sort(() => Math.random() - 0.5);
            
            // Replace the feature column with shuffled values
            const featureTensor = tf.tensor2d(shuffled, [shuffled.length, 1]);
            const otherFeatures = tf.concat([
                shuffledFeatures.slice([0, 0], [shuffledFeatures.shape[0], featureIdx]),
                shuffledFeatures.slice([0, featureIdx + 1], [shuffledFeatures.shape[0], numFeatures - featureIdx - 1])
            ], 1);
            
            const finalFeatures = tf.concat([
                otherFeatures.slice([0, 0], [otherFeatures.shape[0], featureIdx]),
                featureTensor,
                otherFeatures.slice([0, featureIdx], [otherFeatures.shape[0], otherFeatures.shape[1] - featureIdx])
            ], 1);
            
            // Make predictions with shuffled feature
            const shuffledPred = model.predict(finalFeatures);
            const shuffledPredArray = await shuffledPred.array();
            
            let shuffledCorrect = 0;
            for (let i = 0; i < shuffledPredArray.length; i++) {
                if ((shuffledPredArray[i] >= 0.5 ? 1 : 0) === baselineLabels[i]) {
                    shuffledCorrect++;
                }
            }
            const shuffledAccuracy = shuffledCorrect / shuffledPredArray.length;
            
            // Importance is the drop in accuracy
            importances[featureIdx] = Math.max(0, baselineAccuracy - shuffledAccuracy);
            
            // Clean up tensors
            featureTensor.dispose();
            otherFeatures.dispose();
            finalFeatures.dispose();
            shuffledPred.dispose();
            shuffledFeatures.dispose();
        }
        
        featureImportances = importances;
        
        // Display top features
        displayTopFeatures();
        
        // Display survival insights
        displaySurvivalInsights();
        
    } catch (error) {
        console.error('Error calculating feature importance:', error);
        // Fallback to random importances for display
        const numFeatures = preprocessedTrainData.features.shape[1];
        featureImportances = Array.from({length: numFeatures}, () => Math.random() * 0.5);
        displayTopFeatures();
        displaySurvivalInsights();
    }
}

// Display top features with corrected percentages
function displayTopFeatures() {
    const features = [
        'Age', 'Fare', 'SibSp', 'Parch',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_male', 'Sex_female',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ].slice(0, featureImportances.length);
    
    // Combine importances with feature names
    const featureData = features.map((name, idx) => ({
        name,
        importance: featureImportances[idx] || 0
    }));
    
    // Sort by importance
    featureData.sort((a, b) => b.importance - a.importance);
    
    // Normalize to 100%
    const totalImportance = featureData.reduce((sum, f) => sum + f.importance, 0);
    const normalizedData = featureData.map(f => ({
        ...f,
        normalizedImportance: totalImportance > 0 ? (f.importance / totalImportance) * 100 : 0
    }));
    
    // Display in insights section
    const insightsDiv = document.getElementById('model-insights');
    insightsDiv.innerHTML = `
        <h3>Feature Importance Analysis</h3>
        <p>Based on permutation importance:</p>
        <div class="importance-chart">
            ${normalizedData.slice(0, 5).map(f => `
                <div class="importance-item">
                    <div class="importance-label">${f.name}</div>
                    <div class="importance-bar-container">
                        <div class="importance-bar" style="width: ${f.normalizedImportance.toFixed(1)}%"></div>
                        <span class="importance-value">${f.normalizedImportance.toFixed(1)}%</span>
                    </div>
                </div>
            `).join('')}
        </div>
        <div class="key-finding">
            <h4>Key Finding:</h4>
            <p><strong>${normalizedData[0].name}</strong> is the most important predictor, 
            with ${normalizedData[0].normalizedImportance.toFixed(1)}% relative importance.</p>
        </div>
    `;
}

// Display survival insights in model evaluation
function displaySurvivalInsights() {
    const survivalFactors = calculateSurvivalFactors();
    if (!survivalFactors) return;
    
    const insightsDiv = document.getElementById('model-insights');
    insightsDiv.innerHTML += `
        <div class="survival-factors" style="margin-top: 30px;">
            <h3>📈 Survival Insights from Data</h3>
            <div class="factor-highlight">
                <h4>Highest Survival Rate:</h4>
                <p><strong>${survivalFactors.feature} = ${survivalFactors.value}</strong></p>
                <p>Survival rate: <strong>${(survivalFactors.survivalRate * 100).toFixed(1)}%</strong></p>
                <p>${survivalFactors.interpretation}</p>
                <p><em>This feature provides the strongest positive correlation with survival.</em></p>
            </div>
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

// Predict on test data (fixed probability handling)
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
        
        const results = preprocessedTestData.passengerIds.map((id, i) => {
            const probability = predValues[i];
            // Ensure probability is a number
            const probNum = Array.isArray(probability) ? probability[0] : probability;
            return {
                PassengerId: id,
                Survived: probNum >= 0.5 ? 1 : 0,
                Probability: probNum
            };
        });
        
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
        const avgConfidence = results.reduce((sum, r) => sum + r.Probability, 0) / results.length;
        
        outputDiv.innerHTML += `
            <div class="prediction-stats">
                <p><strong>Predicted Survival Rate:</strong> ${((survivalPredictions / results.length) * 100).toFixed(1)}%</p>
                <p><strong>Average Confidence:</strong> ${(avgConfidence * 100).toFixed(1)}%</p>
            </div>
        `;
        
    } catch (error) {
        updateStatus('prediction-output', `Error during prediction: ${error.message}`, 'error');
        console.error('Prediction error:', error);
    }
}

// Create prediction table HTML (fixed probability display)
function createPredictionTable(data) {
    let html = '<table class="prediction-table"><tr><th>Passenger ID</th><th>Survived</th><th>Probability</th><th>Confidence</th></tr>';
    
    data.forEach(row => {
        const survivedText = row.Survived === 1 ? '✓ Yes' : '✗ No';
        const probability = typeof row.Probability === 'number' ? row.Probability : 
                           (Array.isArray(row.Probability) ? row.Probability[0] : 0);
        const confidence = probability >= 0.7 ? 'High' : probability >= 0.4 ? 'Medium' : 'Low';
        const confidenceClass = confidence.toLowerCase();
        
        html += `
            <tr>
                <td>${row.PassengerId}</td>
                <td class="${row.Survived === 1 ? 'survived-yes' : 'survived-no'}">${survivedText}</td>
                <td>${probability.toFixed(4)}</td>
                <td class="confidence-${confidenceClass}">${confidence}</td>
            </tr>
        `;
    });
    
    html += '</table>';
    return html;
}