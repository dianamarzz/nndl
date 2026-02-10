// app.js - Titanic Survival Predictor with Neural Network
// Extends the original EDA dashboard with TensorFlow.js classification

// ============================================================================
// GLOBAL VARIABLES & STATE
// ============================================================================

// Data storage
let trainData = [];
let testData = [];
let mergedData = [];
let allData = []; // For EDA on combined dataset

// TensorFlow.js variables
let model = null;
let trainingHistory = [];
let validationData = null;
let featureNames = [];
let featureImportance = {};
let predictions = null;
let predictionProbabilities = null;

// UI state
let charts = {};
let columnStats = {};
let preprocessingComplete = false;
let modelCreated = false;
let modelTrained = false;

// ============================================================================
// INITIALIZATION & EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic Survival Predictor initialized');
    
    // Initialize event listeners
    document.getElementById('loadDataBtn').addEventListener('click', loadAndMergeData);
    document.getElementById('runEDABtn').addEventListener('click', runEDA);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('createModelBtn').addEventListener('click', createModel);
    document.getElementById('trainModelBtn').addEventListener('click', trainModel);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', makePredictions);
    document.getElementById('exportSubmissionBtn').addEventListener('click', exportSubmission);
    document.getElementById('exportProbabilitiesBtn').addEventListener('click', exportProbabilities);
    document.getElementById('saveModelBtn').addEventListener('click', saveModel);
    document.getElementById('thresholdSlider').addEventListener('input', updateThreshold);
    
    // Initialize TensorFlow.js backend info
    console.log('TensorFlow.js backend:', tf.getBackend());
});

// ============================================================================
// DATA LOADING & MERGING
// ============================================================================

function loadAndMergeData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please select both train.csv and test.csv files from Kaggle Titanic dataset.');
        return;
    }
    
    // Show loading state
    const loadBtn = document.getElementById('loadDataBtn');
    const buttonText = document.getElementById('loadButtonText');
    const spinner = document.getElementById('loadSpinner');
    
    buttonText.textContent = 'Loading...';
    spinner.style.display = 'inline-block';
    loadBtn.disabled = true;
    
    // Parse files sequentially to avoid issues
    parseCSV(trainFile)
        .then(trainResults => {
            if (trainResults.errors.length > 0) {
                throw new Error('Train CSV error: ' + trainResults.errors[0].message);
            }
            trainData = trainResults.data;
            trainData.forEach(row => row._isTrain = true);
            
            return parseCSV(testFile);
        })
        .then(testResults => {
            if (testResults.errors.length > 0) {
                throw new Error('Test CSV error: ' + testResults.errors[0].message);
            }
            testData = testResults.data;
            testData.forEach(row => row._isTrain = false);
            
            // Merge for EDA
            mergedData = [...trainData, ...testData];
            allData = mergedData; // Alias for EDA functions
            
            console.log(`Loaded ${trainData.length} training rows and ${testData.length} test rows`);
            
            // Update UI
            updateDatasetOverview();
            
            // Show dataset overview section
            document.getElementById('datasetOverview').style.display = 'block';
            
            // Reset button
            buttonText.textContent = '📁 Load & Merge Datasets';
            spinner.style.display = 'none';
            loadBtn.disabled = false;
            
            // Show success message
            alert(`Successfully loaded datasets!\n\nTraining data: ${trainData.length} passengers\nTest data: ${testData.length} passengers\nTotal: ${mergedData.length} passengers`);
            
        })
        .catch(error => {
            alert('Error loading files: ' + error.message);
            console.error('Load error:', error);
            
            // Reset button
            buttonText.textContent = '📁 Load & Merge Datasets';
            spinner.style.display = 'none';
            loadBtn.disabled = false;
        });
}

// Helper function to parse CSV
function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: resolve,
            error: reject
        });
    });
}

function updateDatasetOverview() {
    document.getElementById('totalRows').textContent = mergedData.length.toLocaleString();
    document.getElementById('trainRows').textContent = trainData.length.toLocaleString();
    document.getElementById('testRows').textContent = testData.length.toLocaleString();
    
    // Count features (excluding PassengerId, Survived, and internal flags)
    const sampleRow = mergedData[0] || {};
    const excludedCols = ['PassengerId', 'Survived', '_isTrain'];
    const featureCols = Object.keys(sampleRow).filter(col => !excludedCols.includes(col));
    document.getElementById('featureCount').textContent = featureCols.length;
}

// ============================================================================
// EDA FUNCTIONS (adapted from original)
// ============================================================================

function runEDA() {
    console.log('Running EDA on merged dataset...');
    
    // Show all EDA sections
    for (let i = 2; i <= 7; i++) {
        const section = document.getElementById(['dataOverviewSection', 'dataTypesSection', 
            'distributionsSection', 'outliersSection', 'correlationSection', 
            'survivalSection'][i-2]);
        if (section) section.style.display = 'block';
    }
    
    // Run EDA pipeline (simplified version)
    performEDA();
}

// Main EDA pipeline (adapted from original)
function performEDA() {
    console.log('Starting EDA pipeline...');
    
    // 1. Data Overview
    displayDataOverview();
    
    // 2. Data Types & Missing Values
    analyzeDataTypesAndMissing();
    
    // 3. Distribution Analysis
    createDistributionCharts();
    
    // Note: Outlier, correlation, and survival analysis sections
    // are populated when user clicks the respective tab buttons
    
    console.log('EDA pipeline completed');
}

// 1. Display Data Overview
function displayDataOverview() {
    try {
        const container = document.getElementById('edaOverviewContent');
        const rowCount = mergedData.length;
        const colCount = Object.keys(mergedData[0] || {}).length;
        
        let html = `
            <div class="row mb-3">
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-1 text-muted">Total Rows</h6>
                            <h4 class="card-title">${rowCount.toLocaleString()}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-1 text-muted">Columns</h6>
                            <h4 class="card-title">${colCount}</h4>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-1 text-muted">Train Survival Rate</h6>
        `;
        
        // Calculate survival rate from training data only
        const survivedCount = trainData.filter(row => row.Survived === 1).length;
        const survivalRate = trainData.length > 0 ? ((survivedCount / trainData.length) * 100).toFixed(1) : '0.0';
        html += `<h4 class="card-title">${survivalRate}%</h4>`;
        
        html += `
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card stat-card">
                        <div class="card-body">
                            <h6 class="card-subtitle mb-1 text-muted">Complete Cases</h6>
        `;
        
        const completeCases = mergedData.filter(row => {
            return Object.values(row).every(value => value !== null && value !== undefined && value !== '');
        }).length;
        const completeCasesPct = ((completeCases / rowCount) * 100).toFixed(1);
        html += `<h4 class="card-title">${completeCasesPct}%</h4>`;
        
        html += `
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="table-responsive">
                <h6 class="mb-2">Data Preview (First 5 Rows from each dataset)</h6>
                <table class="table table-sm table-hover data-table">
                    <thead>
                        <tr>
        `;
        
        // Get column names
        const columns = Object.keys(mergedData[0] || {});
        columns.forEach(col => {
            if (col !== '_isTrain') {
                html += `<th>${col}</th>`;
            }
        });
        
        html += `</tr></thead><tbody>`;
        
        // Show first 5 rows from train and test
        const previewRows = [...trainData.slice(0, 5), ...testData.slice(0, 5)];
        previewRows.forEach((row, idx) => {
            html += `<tr>`;
            columns.forEach(col => {
                if (col !== '_isTrain') {
                    let value = row[col];
                    if (value === null || value === undefined || value === '') {
                        value = '<span class="text-muted">-</span>';
                        html += `<td>${value}</td>`;
                    } else if (typeof value === 'number') {
                        html += `<td>${Number.isInteger(value) ? value : value.toFixed(2)}</td>`;
                    } else {
                        html += `<td>${value}</td>`;
                    }
                }
            });
            html += `</tr>`;
        });
        
        html += `</tbody></table>`;
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error in displayDataOverview:', error);
        document.getElementById('edaOverviewContent').innerHTML = 
            `<div class="alert alert-danger">Error displaying data overview: ${error.message}</div>`;
    }
}

// 2. Analyze Data Types and Missing Values
function analyzeDataTypesAndMissing() {
    try {
        const container = document.getElementById('edaTypesContent');
        let html = `<div class="row"><div class="col-md-6"><h6>Column Information</h6>`;
        
        html += `<div class="table-responsive"><table class="table table-sm table-hover data-table">
            <thead><tr><th>Column</th><th>Data Type</th><th>Missing Values</th><th>Missing %</th></tr></thead><tbody>`;
        
        const columns = Object.keys(mergedData[0] || {}).filter(col => col !== '_isTrain');
        columnStats = {};
        
        columns.forEach(col => {
            const nonMissingCount = mergedData.filter(row => {
                const value = row[col];
                return value !== null && value !== undefined && value !== '';
            }).length;
            
            const missingCount = mergedData.length - nonMissingCount;
            const missingPct = ((missingCount / mergedData.length) * 100).toFixed(1);
            
            // Determine data type
            const sampleRow = mergedData.find(row => row[col] !== null && row[col] !== undefined && row[col] !== '');
            let dataType = 'unknown';
            if (sampleRow) {
                dataType = typeof sampleRow[col];
                if (dataType === 'number' && ['Pclass', 'SibSp', 'Parch', 'Survived'].includes(col)) {
                    dataType = 'integer (categorical)';
                } else if (dataType === 'number') {
                    dataType = 'float';
                } else if (dataType === 'string') {
                    dataType = 'text';
                }
            }
            
            // Store column stats for later use
            columnStats[col] = {
                missingCount,
                missingPct,
                dataType,
                nonMissingCount
            };
            
            html += `<tr>
                <td><strong>${col}</strong></td>
                <td><span class="badge bg-secondary">${dataType}</span></td>
                <td>${missingCount}</td>
                <td>
                    <div class="progress" style="height: 8px;">
                        <div class="progress-bar bg-warning" role="progressbar" 
                             style="width: ${missingPct}%" 
                             aria-valuenow="${missingPct}" 
                             aria-valuemin="0" 
                             aria-valuemax="100"></div>
                    </div>
                    <span class="small">${missingPct}%</span>
                </td>
            </tr>`;
        });
        
        html += `</tbody></table></div></div><div class="col-md-6"><h6>Missing Values Visualization</h6>`;
        html += `<div class="chart-container"><canvas id="missingValuesChart"></canvas></div>`;
        
        html += `</div></div><div class="mt-3"><div class="alert alert-info">`;
        html += `<strong>Missing Data Summary:</strong> Age (${columnStats['Age']?.missingPct || 0}%) and Cabin (${columnStats['Cabin']?.missingPct || 0}%) have the most missing values.`;
        html += `</div></div>`;
        
        container.innerHTML = html;
        
        // Create missing values chart
        createMissingValuesChart();
        
    } catch (error) {
        console.error('Error in analyzeDataTypesAndMissing:', error);
        document.getElementById('edaTypesContent').innerHTML = 
            `<div class="alert alert-danger">Error analyzing data types: ${error.message}</div>`;
    }
}

// Create missing values chart
function createMissingValuesChart() {
    try {
        const ctx = document.getElementById('missingValuesChart');
        if (!ctx) return;
        
        if (charts.missingValues) {
            charts.missingValues.destroy();
        }
        
        const columns = Object.keys(columnStats).sort((a, b) => 
            parseFloat(columnStats[b].missingPct) - parseFloat(columnStats[a].missingPct)
        );
        
        const labels = columns;
        const missingData = columns.map(col => parseFloat(columnStats[col].missingPct) || 0);
        
        charts.missingValues = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Missing Values %',
                    data: missingData,
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgb(52, 152, 219)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Percentage Missing' }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error in createMissingValuesChart:', error);
    }
}

// 3. Create Distribution Charts
function createDistributionCharts() {
    try {
        const container = document.getElementById('edaDistributionsContent');
        
        let html = `<ul class="nav nav-tabs" id="distTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="num-tab" data-bs-toggle="tab" data-bs-target="#num-dist" type="button" role="tab">Numerical</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="cat-tab" data-bs-toggle="tab" data-bs-target="#cat-dist" type="button" role="tab">Categorical</button>
            </li>
        </ul>
        <div class="tab-content p-3 border border-top-0 rounded-bottom">
            <div class="tab-pane fade show active" id="num-dist" role="tabpanel">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Age Distribution</h6>
                        <div class="chart-container">
                            <canvas id="ageDistChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Fare Distribution</h6>
                        <div class="chart-container">
                            <canvas id="fareDistChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-6">
                        <h6>SibSp Distribution</h6>
                        <div class="chart-container">
                            <canvas id="sibspDistChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Parch Distribution</h6>
                        <div class="chart-container">
                            <canvas id="parchDistChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="cat-dist" role="tabpanel">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Gender Distribution</h6>
                        <div class="chart-container">
                            <canvas id="sexDistChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Passenger Class Distribution</h6>
                        <div class="chart-container">
                            <canvas id="pclassDistChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-6">
                        <h6>Embarkation Port Distribution</h6>
                        <div class="chart-container">
                            <canvas id="embarkedDistChart"></canvas>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h6>Survival Distribution (Train only)</h6>
                        <div class="chart-container">
                            <canvas id="survivedDistChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        
        container.innerHTML = html;
        
        // Create charts
        createNumericalDistributionCharts();
        createCategoricalDistributionCharts();
        
        // Reinitialize Bootstrap tabs
        const tabEl = document.querySelector('#distTab button[data-bs-target="#num-dist"]');
        if (tabEl) {
            new bootstrap.Tab(tabEl).show();
        }
        
    } catch (error) {
        console.error('Error in createDistributionCharts:', error);
    }
}

// Helper function for numerical distributions
function createNumericalDistributionCharts() {
    try {
        // Age Distribution
        const ageData = mergedData
            .filter(row => row.Age !== null && !isNaN(row.Age))
            .map(row => row.Age);
        
        if (ageData.length > 0) {
            createHistogram('ageDistChart', ageData, 'Age', 'Passenger Count', 'Age Distribution');
        }
        
        // Fare Distribution
        const fareData = mergedData
            .filter(row => row.Fare !== null && !isNaN(row.Fare))
            .map(row => row.Fare);
        
        if (fareData.length > 0) {
            createHistogram('fareDistChart', fareData, 'Fare', 'Passenger Count', 'Fare Distribution', 20);
        }
        
        // SibSp Distribution
        const sibspData = mergedData
            .filter(row => row.SibSp !== null && !isNaN(row.SibSp))
            .map(row => row.SibSp);
        
        if (sibspData.length > 0) {
            createHistogram('sibspDistChart', sibspData, 'Siblings/Spouses', 'Passenger Count', 'SibSp Distribution', 7);
        }
        
        // Parch Distribution
        const parchData = mergedData
            .filter(row => row.Parch !== null && !isNaN(row.Parch))
            .map(row => row.Parch);
        
        if (parchData.length > 0) {
            createHistogram('parchDistChart', parchData, 'Parents/Children', 'Passenger Count', 'Parch Distribution', 7);
        }
    } catch (error) {
        console.error('Error in createNumericalDistributionCharts:', error);
    }
}

// Helper function for categorical distributions
function createCategoricalDistributionCharts() {
    try {
        // Gender Distribution
        const sexCounts = countCategories('Sex');
        if (Object.keys(sexCounts).length > 0) {
            createBarChart('sexDistChart', 
                Object.keys(sexCounts), 
                Object.values(sexCounts), 
                'Gender', 
                'Count', 
                'Gender Distribution',
                ['rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
            );
        }
        
        // Passenger Class Distribution
        const pclassCounts = countCategories('Pclass');
        if (Object.keys(pclassCounts).length > 0) {
            createBarChart('pclassDistChart', 
                Object.keys(pclassCounts).map(key => `Class ${key}`), 
                Object.values(pclassCounts), 
                'Passenger Class', 
                'Count', 
                'Passenger Class Distribution',
                ['rgba(46, 204, 113, 0.7)', 'rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
            );
        }
        
        // Embarked Port Distribution
        const embarkedCounts = countCategories('Embarked');
        if (Object.keys(embarkedCounts).length > 0) {
            createBarChart('embarkedDistChart', 
                Object.keys(embarkedCounts).map(key => {
                    if (key === 'C') return 'Cherbourg';
                    if (key === 'Q') return 'Queenstown';
                    if (key === 'S') return 'Southampton';
                    return key;
                }), 
                Object.values(embarkedCounts), 
                'Embarkation Port', 
                'Count', 
                'Embarkation Port Distribution',
                ['rgba(52, 152, 219, 0.7)', 'rgba(46, 204, 113, 0.7)', 'rgba(241, 196, 15, 0.7)']
            );
        }
        
        // Survival Distribution (train only)
        const survivedCounts = countCategoriesForTrain('Survived');
        if (Object.keys(survivedCounts).length > 0) {
            createBarChart('survivedDistChart', 
                ['Perished', 'Survived'], 
                [survivedCounts[0] || 0, survivedCounts[1] || 0], 
                'Outcome', 
                'Count', 
                'Survival Distribution (Training Data)',
                ['rgba(231, 76, 60, 0.7)', 'rgba(46, 204, 113, 0.7)']
            );
        }
    } catch (error) {
        console.error('Error in createCategoricalDistributionCharts:', error);
    }
}

// Helper function to count categories
function countCategories(columnName) {
    const counts = {};
    mergedData.forEach(row => {
        const value = row[columnName];
        if (value !== null && value !== undefined && value !== '') {
            counts[value] = (counts[value] || 0) + 1;
        }
    });
    return counts;
}

// Helper function to count categories in training data only
function countCategoriesForTrain(columnName) {
    const counts = {};
    trainData.forEach(row => {
        const value = row[columnName];
        if (value !== null && value !== undefined && value !== '') {
            counts[value] = (counts[value] || 0) + 1;
        }
    });
    return counts;
}

// Helper function to create histogram (from original)
function createHistogram(canvasId, data, xLabel, yLabel, title, bins = 15) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        
        const min = Math.min(...data);
        const max = Math.max(...data);
        const binWidth = (max - min) / bins;
        
        const binCounts = new Array(bins).fill(0);
        data.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
            binCounts[binIndex]++;
        });
        
        const binLabels = [];
        for (let i = 0; i < bins; i++) {
            const binStart = min + (i * binWidth);
            const binEnd = min + ((i + 1) * binWidth);
            binLabels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
        }
        
        charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: yLabel,
                    data: binCounts,
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgb(52, 152, 219)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: xLabel },
                        ticks: { maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: yLabel },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error(`Error in createHistogram for ${canvasId}:`, error);
    }
}

// Helper function to create bar chart (from original)
function createBarChart(canvasId, labels, data, xLabel, yLabel, title, colors = null) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        
        const backgroundColors = colors || labels.map((_, i) => 
            `hsl(${(i * 360) / labels.length}, 70%, 65%)`
        );
        
        charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: yLabel,
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: { display: true, text: title },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: xLabel }
                    },
                    y: {
                        title: { display: true, text: yLabel },
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error(`Error in createBarChart for ${canvasId}:`, error);
    }
}

// ============================================================================
// PREPROCESSING FOR NEURAL NETWORK
// ============================================================================

async function preprocessData() {
    console.log('Starting data preprocessing...');
    
    // Show loading state
    const buttonText = document.getElementById('trainButtonText');
    buttonText.textContent = 'Preprocessing...';
    
    try {
        // ============================================================================
        // DATA SCHEMA DEFINITION
        // ============================================================================
        // TARGET: Survived (0/1)
        // FEATURES: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
        // EXCLUDE: PassengerId (use only for submission), Name, Ticket, Cabin
        // DERIVED: FamilySize, IsAlone
        
        // ============================================================================
        // STEP 1: Prepare training data
        // ============================================================================
        console.log('Preparing training data...');
        
        // Filter out rows with missing target
        const validTrainData = trainData.filter(row => 
            row.Survived !== null && row.Survived !== undefined
        );
        
        // Extract features and labels
        const rawFeatures = [];
        const labels = [];
        const passengerIds = [];
        
        validTrainData.forEach(row => {
            // Store PassengerId for later reference
            passengerIds.push(row.PassengerId);
            
            // Extract features based on schema
            const features = [
                row.Pclass || 3,                     // 1st, 2nd, 3rd class
                row.Sex === 'female' ? 1 : 0,        // Female = 1, Male = 0
                row.Age || 0,                        // Age (will be imputed)
                row.SibSp || 0,                      # Siblings/Spouses
                row.Parch || 0,                      # Parents/Children
                row.Fare || 0,                       # Fare (will be standardized)
                row.Embarked === 'C' ? 1 : 0,        # Cherbourg
                row.Embarked === 'Q' ? 1 : 0,        # Queenstown
                row.Embarked === 'S' ? 1 : 0,        # Southampton
                (row.SibSp || 0) + (row.Parch || 0) + 1,  # FamilySize
                ((row.SibSp || 0) + (row.Parch || 0)) === 0 ? 1 : 0  # IsAlone
            ];
            
            rawFeatures.push(features);
            labels.push(row.Survived);
        });
        
        // ============================================================================
        // STEP 2: Impute missing values (Age)
        // ============================================================================
        console.log('Imputing missing values...');
        
        // Calculate median Age from non-missing values
        const ages = validTrainData.map(row => row.Age).filter(age => age && !isNaN(age));
        const medianAge = ages.length > 0 ? 
            ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 29;
        
        console.log(`Median Age for imputation: ${medianAge}`);
        
        // Impute Age in features
        const featuresWithImputedAge = rawFeatures.map(features => {
            const newFeatures = [...features];
            // Age is at index 2
            if (!features[2] || features[2] === 0) {
                newFeatures[2] = medianAge;
            }
            return newFeatures;
        });
        
        // ============================================================================
        // STEP 3: Standardize numerical features (Age and Fare)
        // ============================================================================
        console.log('Standardizing numerical features...');
        
        // Calculate mean and std for Age
        const ageValues = featuresWithImputedAge.map(f => f[2]);
        const ageMean = ageValues.reduce((a, b) => a + b, 0) / ageValues.length;
        const ageStd = Math.sqrt(
            ageValues.map(x => Math.pow(x - ageMean, 2)).reduce((a, b) => a + b, 0) / ageValues.length
        ) || 1;
        
        // Calculate mean and std for Fare
        const fareValues = featuresWithImputedAge.map(f => f[5]);
        const fareMean = fareValues.reduce((a, b) => a + b, 0) / fareValues.length;
        const fareStd = Math.sqrt(
            fareValues.map(x => Math.pow(x - fareMean, 2)).reduce((a, b) => a + b, 0) / fareValues.length
        ) || 1;
        
        // Apply standardization
        const processedFeatures = featuresWithImputedAge.map(features => {
            const newFeatures = [...features];
            // Standardize Age (index 2)
            newFeatures[2] = (newFeatures[2] - ageMean) / ageStd;
            // Standardize Fare (index 5)
            newFeatures[5] = (newFeatures[5] - fareMean) / fareStd;
            return newFeatures;
        });
        
        // ============================================================================
        // STEP 4: Store preprocessing parameters for test data
        // ============================================================================
        window.preprocessingParams = {
            medianAge,
            ageMean,
            ageStd,
            fareMean,
            fareStd
        };
        
        // ============================================================================
        // STEP 5: Create feature names for display
        // ============================================================================
        featureNames = [
            'Pclass', 'Sex_female', 'Age_std', 'SibSp', 'Parch', 'Fare_std',
            'Embarked_C', 'Embarked_Q', 'Embarked_S', 'FamilySize', 'IsAlone'
        ];
        
        // ============================================================================
        // STEP 6: Convert to tensors
        // ============================================================================
        console.log('Converting to tensors...');
        
        // Convert features to tensor
        const featureTensor = tf.tensor2d(processedFeatures, [processedFeatures.length, processedFeatures[0].length]);
        
        // Convert labels to tensor (reshape for binary classification)
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
        
        // ============================================================================
        // STEP 7: Create train/validation split (80/20 stratified)
        // ============================================================================
        console.log('Creating train/validation split...');
        
        // Simple random split (for simplicity)
        const totalSamples = processedFeatures.length;
        const trainSize = Math.floor(totalSamples * 0.8);
        
        // Create indices and shuffle
        const indices = tf.util.createShuffledIndices(totalSamples);
        
        // Split indices
        const trainIndices = indices.slice(0, trainSize);
        const valIndices = indices.slice(trainSize);
        
        // Create train tensors
        const trainFeatures = tf.gather(featureTensor, trainIndices);
        const trainLabels = tf.gather(labelTensor, trainIndices);
        
        // Create validation tensors
        const valFeatures = tf.gather(featureTensor, valIndices);
        const valLabels = tf.gather(labelTensor, valIndices);
        
        // Store validation data for later evaluation
        validationData = { features: valFeatures, labels: valLabels };
        
        // ============================================================================
        // STEP 8: Store processed data for model training
        // ============================================================================
        window.trainingData = {
            features: trainFeatures,
            labels: trainLabels,
            featureNames: featureNames,
            preprocessingParams: window.preprocessingParams
        };
        
        // ============================================================================
        // STEP 9: Update UI
        // ============================================================================
        preprocessingComplete = true;
        
        document.getElementById('inputNeurons').textContent = processedFeatures[0].length;
        document.getElementById('modelSection').style.display = 'block';
        document.getElementById('trainingSection').style.display = 'block';
        
        buttonText.textContent = '🚀 Train Neural Network';
        
        console.log(`Preprocessing complete. Features: ${processedFeatures[0].length}, Samples: ${processedFeatures.length}`);
        console.log('Feature names:', featureNames);
        
        // Show success message
        alert(`Preprocessing complete!\n\nFeatures: ${featureNames.length}\nTraining samples: ${trainSize}\nValidation samples: ${totalSamples - trainSize}\n\nClick "Create Model" to build the neural network architecture.`);
        
    } catch (error) {
        console.error('Error in preprocessing:', error);
        alert('Error during preprocessing: ' + error.message);
        buttonText.textContent = '🚀 Train Neural Network';
    }
}

// ============================================================================
// NEURAL NETWORK MODEL
// ============================================================================

function createModel() {
    if (!preprocessingComplete) {
        alert('Please preprocess data first by clicking "Preprocess & Train Model"');
        return;
    }
    
    console.log('Creating neural network model...');
    
    try {
        // Clear any existing model
        if (model) {
            model.dispose();
        }
        
        // ============================================================================
        // MODEL ARCHITECTURE
        // ============================================================================
        // Shallow neural network with:
        // - Input layer: nFeatures neurons
        // - Hidden layer: 16 neurons, ReLU activation
        // - Output layer: 1 neuron, Sigmoid activation (binary classification)
        
        const nFeatures = window.trainingData.features.shape[1];
        
        model = tf.sequential();
        
        // Hidden layer
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [nFeatures],
            kernelInitializer: 'glorotNormal',
            name: 'hidden_layer'
        }));
        
        // Output layer (binary classification)
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }));
        
        // ============================================================================
        // MODEL COMPILATION
        // ============================================================================
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // ============================================================================
        // DISPLAY MODEL SUMMARY
        // ============================================================================
        let summaryText = 'Model: "sequential"\n';
        summaryText += '_________________________________________________________________\n';
        
        model.summary(null, null, (line) => {
            summaryText += line + '\n';
        });
        
        // Wait a moment for summary to populate
        setTimeout(() => {
            document.getElementById('modelSummary').textContent = summaryText;
            
            // Count parameters
            const paramCount = model.countParams();
            document.getElementById('paramCount').textContent = paramCount.toLocaleString();
            
            // Update UI
            modelCreated = true;
            document.getElementById('createModelBtn').textContent = 'Model Created ✓';
            document.getElementById('createModelBtn').classList.remove('btn-primary');
            document.getElementById('createModelBtn').classList.add('btn-success');
            document.getElementById('createModelBtn').disabled = true;
            
            console.log('Model created successfully');
            console.log(`Parameters: ${paramCount}`);
            
            // Show training button as enabled
            document.getElementById('trainModelBtn').disabled = false;
            
        }, 100);
        
    } catch (error) {
        console.error('Error creating model:', error);
        alert('Error creating model: ' + error.message);
    }
}

async function trainModel() {
    if (!modelCreated) {
        alert('Please create the model first by clicking "Create Model"');
        return;
    }
    
    console.log('Starting model training...');
    
    // Show loading state
    const trainBtn = document.getElementById('trainModelBtn');
    const buttonText = document.getElementById('trainButtonText');
    const spinner = document.getElementById('trainSpinner');
    
    buttonText.textContent = 'Training...';
    spinner.style.display = 'inline-block';
    trainBtn.disabled = true;
    
    try {
        const { features, labels } = window.trainingData;
        const { features: valFeatures, labels: valLabels } = validationData;
        
        // ============================================================================
        // TRAINING CONFIGURATION
        // ============================================================================
        const epochs = 50;
        const batchSize = 32;
        const patience = 5; // Early stopping patience
        
        // ============================================================================
        // TRAINING WITH TENSORFLOW.JS
        // ============================================================================
        console.log(`Training for ${epochs} epochs with batch size ${batchSize}...`);
        
        // Clear previous charts
        const trainingCharts = document.getElementById('trainingCharts');
        trainingCharts.innerHTML = '<div id="tfjs-vis-charts"></div>';
        
        // Prepare tfjs-vis surface
        const surface = { name: 'Training Metrics', tab: 'Training' };
        
        // Train model with callbacks for visualization
        const history = await model.fit(features, labels, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: [
                // Show training progress with tfjs-vis
                tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc'], {
                    callbacks: ['onEpochEnd']
                }),
                // Early stopping callback
                {
                    onEpochEnd: async (epoch, logs) => {
                        // Store history for later use
                        trainingHistory.push({
                            epoch: epoch + 1,
                            loss: logs.loss,
                            val_loss: logs.val_loss,
                            acc: logs.acc,
                            val_acc: logs.val_acc
                        });
                        
                        // Simple early stopping logic
                        if (epoch > patience) {
                            const recentValLosses = trainingHistory
                                .slice(-patience - 1)
                                .map(h => h.val_loss);
                            
                            const minValLoss = Math.min(...recentValLosses.slice(0, -1));
                            const currentValLoss = recentValLosses[recentValLosses.length - 1];
                            
                            if (currentValLoss > minValLoss) {
                                console.log(`Early stopping triggered at epoch ${epoch + 1}`);
                                model.stopTraining = true;
                            }
                        }
                    }
                }
            ]
        });
        
        // ============================================================================
        // POST-TRAINING
        // ============================================================================
        modelTrained = true;
        console.log('Training completed');
        
        // Calculate feature importance
        await calculateFeatureImportance();
        
        // Update UI
        buttonText.textContent = 'Training Complete ✓';
        spinner.style.display = 'none';
        trainBtn.classList.remove('btn-primary');
        trainBtn.classList.add('btn-success');
        
        // Show evaluate button
        document.getElementById('evaluateBtn').style.display = 'inline-block';
        document.getElementById('predictionSection').style.display = 'block';
        
        // Show success message
        const finalEpoch = history.epoch.length;
        const finalValAcc = history.history.val_acc[finalEpoch - 1];
        alert(`Training completed!\n\nEpochs: ${finalEpoch}\nValidation Accuracy: ${(finalValAcc * 100).toFixed(1)}%`);
        
    } catch (error) {
        console.error('Error during training:', error);
        alert('Error during training: ' + error.message);
        
        // Reset button
        buttonText.textContent = '🚀 Train Neural Network';
        spinner.style.display = 'none';
        trainBtn.disabled = false;
    }
}

// ============================================================================
// FEATURE IMPORTANCE CALCULATION
// ============================================================================

async function calculateFeatureImportance() {
    if (!modelTrained || !validationData) return;
    
    console.log('Calculating feature importance...');
    
    try {
        const { features, labels } = validationData;
        const baselineData = await features.array();
        const baselineLabels = await labels.array();
        
        // Get baseline accuracy
        const predictions = model.predict(features);
        const predictedClasses = (await predictions.array()).map(p => p[0] > 0.5 ? 1 : 0);
        
        let baselineCorrect = 0;
        for (let i = 0; i < baselineLabels.length; i++) {
            if (predictedClasses[i] === baselineLabels[i][0]) {
                baselineCorrect++;
            }
        }
        const baselineAccuracy = baselineCorrect / baselineLabels.length;
        
        // Calculate permutation importance for each feature
        featureImportance = {};
        
        for (let featureIdx = 0; featureIdx < featureNames.length; featureIdx++) {
            // Create shuffled copy of the feature
            const shuffledData = JSON.parse(JSON.stringify(baselineData));
            const featureValues = shuffledData.map(row => row[featureIdx]);
            
            // Shuffle the feature values
            for (let i = featureValues.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [featureValues[i], featureValues[j]] = [featureValues[j], featureValues[i]];
            }
            
            // Apply shuffled values
            shuffledData.forEach((row, idx) => {
                row[featureIdx] = featureValues[idx];
            });
            
            // Make predictions with shuffled feature
            const shuffledTensor = tf.tensor2d(shuffledData);
            const shuffledPredictions = model.predict(shuffledTensor);
            const shuffledClasses = (await shuffledPredictions.array()).map(p => p[0] > 0.5 ? 1 : 0);
            
            // Calculate accuracy with shuffled feature
            let shuffledCorrect = 0;
            for (let i = 0; i < baselineLabels.length; i++) {
                if (shuffledClasses[i] === baselineLabels[i][0]) {
                    shuffledCorrect++;
                }
            }
            const shuffledAccuracy = shuffledCorrect / baselineLabels.length;
            
            // Importance = decrease in accuracy
            const importance = baselineAccuracy - shuffledAccuracy;
            featureImportance[featureNames[featureIdx]] = importance;
            
            // Clean up tensors
            shuffledTensor.dispose();
            shuffledPredictions.dispose();
        }
        
        // Clean up
        predictions.dispose();
        
        // Display feature importance
        displayFeatureImportance();
        
    } catch (error) {
        console.error('Error calculating feature importance:', error);
    }
}

function displayFeatureImportance() {
    try {
        // Sort features by importance (absolute value)
        const sortedFeatures = Object.entries(featureImportance)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        
        // Find top factor
        const topFeature = sortedFeatures[0];
        document.getElementById('topFactor').textContent = `${topFeature[0]} (importance: ${topFeature[1].toFixed(4)})`;
        document.getElementById('topFactorDisplay').style.display = 'block';
        
        // Create bar chart
        const ctx = document.getElementById('featureImportanceChart');
        if (charts.featureImportance) {
            charts.featureImportance.destroy();
        }
        
        const labels = sortedFeatures.map(f => f[0]);
        const importanceValues = sortedFeatures.map(f => f[1]);
        
        // Color based on positive/negative importance
        const backgroundColors = importanceValues.map(value => 
            value > 0 ? 'rgba(231, 76, 60, 0.7)' : 'rgba(52, 152, 219, 0.7)'
        );
        
        charts.featureImportance = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: importanceValues,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Feature Importance (Permutation Method)'
                    },
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Decrease in Accuracy when Feature is Shuffled'
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error displaying feature importance:', error);
    }
}

// ============================================================================
// MODEL EVALUATION
// ============================================================================

async function evaluateModel() {
    if (!modelTrained || !validationData) {
        alert('Please train the model first');
        return;
    }
    
    console.log('Evaluating model...');
    
    try {
        const { features, labels } = validationData;
        
        // Make predictions
        const predictions = model.predict(features);
        const probs = await predictions.array();
        const trueLabels = await labels.array();
        
        // Calculate ROC curve data
        const rocData = calculateROCData(probs.map(p => p[0]), trueLabels.map(l => l[0]));
        
        // Plot ROC curve
        plotROCCurve(rocData);
        
        // Calculate metrics at default threshold (0.5)
        updateMetrics(probs.map(p => p[0]), trueLabels.map(l => l[0]), 0.5);
        
        // Update confusion matrix
        updateConfusionMatrix(probs.map(p => p[0]), trueLabels.map(l => l[0]), 0.5);
        
        console.log('Model evaluation complete');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        alert('Error during evaluation: ' + error.message);
    }
}

function calculateROCData(probabilities, trueLabels) {
    // Sort by probability descending
    const sorted = probabilities.map((p, i) => ({ p, label: trueLabels[i] }))
        .sort((a, b) => b.p - a.p);
    
    const thresholds = Array.from(new Set(probabilities)).sort((a, b) => b - a);
    const rocPoints = [];
    
    for (const threshold of thresholds) {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        for (const item of sorted) {
            const prediction = item.p >= threshold ? 1 : 0;
            if (prediction === 1 && item.label === 1) tp++;
            if (prediction === 1 && item.label === 0) fp++;
            if (prediction === 0 && item.label === 0) tn++;
            if (prediction === 0 && item.label === 1) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0; // True Positive Rate (Recall)
        const fpr = fp / (fp + tn) || 0; // False Positive Rate
        
        rocPoints.push({ threshold, tpr, fpr, tp, fp, tn, fn });
    }
    
    // Add points for 0 and 1 thresholds
    rocPoints.push({ threshold: 0, tpr: 1, fpr: 1, tp: 0, fp: 0, tn: 0, fn: 0 });
    rocPoints.push({ threshold: 1, tpr: 0, fpr: 0, tp: 0, fp: 0, tn: 0, fn: 0 });
    
    // Calculate AUC (Area Under Curve) using trapezoidal rule
    rocPoints.sort((a, b) => a.fpr - b.fpr);
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const width = rocPoints[i].fpr - rocPoints[i-1].fpr;
        const height = (rocPoints[i].tpr + rocPoints[i-1].tpr) / 2;
        auc += width * height;
    }
    
    return { points: rocPoints, auc: auc };
}

function plotROCCurve(rocData) {
    const ctx = document.getElementById('rocChart');
    if (charts.rocChart) {
        charts.rocChart.destroy();
    }
    
    const fprValues = rocData.points.map(p => p.fpr);
    const tprValues = rocData.points.map(p => p.tpr);
    
    charts.rocChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            datasets: [{
                label: `ROC Curve (AUC = ${rocData.auc.toFixed(3)})`,
                data: fprValues.map((fpr, i) => ({ x: fpr, y: tprValues[i] })),
                borderColor: 'rgb(52, 152, 219)',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: 'Random Classifier',
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: 'rgb(149, 165, 166)',
                borderDash: [5, 5],
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curve'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = rocData.points[context.dataIndex];
                            return `Threshold: ${point.threshold.toFixed(2)}, TPR: ${point.tpr.toFixed(3)}, FPR: ${point.fpr.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate (Recall)'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function updateThreshold() {
    const threshold = parseFloat(document.getElementById('thresholdSlider').value);
    document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
    
    if (modelTrained && validationData) {
        // Recalculate metrics with new threshold
        const probs = window.currentProbabilities || [];
        const trueLabels = window.currentTrueLabels || [];
        
        if (probs.length > 0 && trueLabels.length > 0) {
            updateMetrics(probs, trueLabels, threshold);
            updateConfusionMatrix(probs, trueLabels, threshold);
        }
    }
}

function updateMetrics(probabilities, trueLabels, threshold) {
    // Store for threshold updates
    window.currentProbabilities = probabilities;
    window.currentTrueLabels = trueLabels;
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
        const prediction = probabilities[i] >= threshold ? 1 : 0;
        const actual = trueLabels[i];
        
        if (prediction === 1 && actual === 1) tp++;
        if (prediction === 1 && actual === 0) fp++;
        if (prediction === 0 && actual === 0) tn++;
        if (prediction === 0 && actual === 1) fn++;
    }
    
    // Calculate metrics
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0; // Same as TPR
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    // Display metrics
    const metricsContainer = document.getElementById('metricsDisplay');
    metricsContainer.innerHTML = `
        <div class="metric-box">
            <div class="metric-value">${accuracy.toFixed(3)}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${precision.toFixed(3)}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${recall.toFixed(3)}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">${f1.toFixed(3)}</div>
            <div class="metric-label">F1 Score</div>
        </div>
    `;
}

function updateConfusionMatrix(probabilities, trueLabels, threshold) {
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
        const prediction = probabilities[i] >= threshold ? 1 : 0;
        const actual = trueLabels[i];
        
        if (prediction === 1 && actual === 1) tp++;
        if (prediction === 1 && actual === 0) fp++;
        if (prediction === 0 && actual === 0) tn++;
        if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update table
    document.getElementById('tn').textContent = tn;
    document.getElementById('fp').textContent = fp;
    document.getElementById('fn').textContent = fn;
    document.getElementById('tp').textContent = tp;
}

// ============================================================================
// PREDICTION & EXPORT
// ============================================================================

async function makePredictions() {
    if (!modelTrained) {
        alert('Please train the model first');
        return;
    }
    
    console.log('Making predictions on test data...');
    
    // Show loading state
    const predictBtn = document.getElementById('predictBtn');
    const buttonText = document.getElementById('predictButtonText');
    const spinner = document.getElementById('predictSpinner');
    
    buttonText.textContent = 'Predicting...';
    spinner.style.display = 'inline-block';
    predictBtn.disabled = true;
    
    try {
        // ============================================================================
        // PREPROCESS TEST DATA
        // ============================================================================
        console.log('Preprocessing test data...');
        
        const processedTestFeatures = [];
        const passengerIds = [];
        
        // Get preprocessing parameters
        const params = window.preprocessingParams;
        
        testData.forEach(row => {
            // Store PassengerId
            passengerIds.push(row.PassengerId);
            
            // Extract and preprocess features (same as training)
            const features = [
                row.Pclass || 3,
                row.Sex === 'female' ? 1 : 0,
                row.Age || params.medianAge, // Use imputed median
                row.SibSp || 0,
                row.Parch || 0,
                row.Fare || 0,
                row.Embarked === 'C' ? 1 : 0,
                row.Embarked === 'Q' ? 1 : 0,
                row.Embarked === 'S' ? 1 : 0,
                (row.SibSp || 0) + (row.Parch || 0) + 1,
                ((row.SibSp || 0) + (row.Parch || 0)) === 0 ? 1 : 0
            ];
            
            // Apply standardization (using training parameters)
            features[2] = (features[2] - params.ageMean) / params.ageStd; // Age
            features[5] = (features[5] - params.fareMean) / params.fareStd; // Fare
            
            processedTestFeatures.push(features);
        });
        
        // ============================================================================
        // MAKE PREDICTIONS
        // ============================================================================
        console.log('Running model predictions...');
        
        // Convert to tensor
        const testTensor = tf.tensor2d(processedTestFeatures);
        
        // Make predictions
        const predictionsTensor = model.predict(testTensor);
        const probabilities = await predictionsTensor.array();
        
        // Get current threshold
        const threshold = parseFloat(document.getElementById('thresholdSlider').value);
        
        // Convert probabilities to binary predictions
        const binaryPredictions = probabilities.map(p => p[0] >= threshold ? 1 : 0);
        
        // Store predictions
        window.predictions = {
            passengerIds: passengerIds,
            probabilities: probabilities.map(p => p[0]),
            binaryPredictions: binaryPredictions,
            threshold: threshold
        };
        
        // Clean up tensors
        testTensor.dispose();
        predictionsTensor.dispose();
        
        // ============================================================================
        // DISPLAY RESULTS
        // ============================================================================
        displayPredictionResults();
        
        // Update UI
        buttonText.textContent = 'Predictions Complete ✓';
        spinner.style.display = 'none';
        predictBtn.classList.remove('btn-primary');
        predictBtn.classList.add('btn-success');
        
        // Show prediction statistics
        const survivedCount = binaryPredictions.filter(p => p === 1).length;
        const survivalRate = (survivedCount / binaryPredictions.length * 100).toFixed(1);
        
        document.getElementById('predictionStats').innerHTML = `
            <strong>Prediction Statistics:</strong><br>
            Total predictions: ${binaryPredictions.length}<br>
            Predicted survived: ${survivedCount} (${survivalRate}%)<br>
            Predicted perished: ${binaryPredictions.length - survivedCount}<br>
            Threshold: ${threshold.toFixed(2)}
        `;
        document.getElementById('predictionStats').style.display = 'block';
        
        console.log(`Predictions complete. ${survivedCount} passengers predicted to survive (${survivalRate}%)`);
        
    } catch (error) {
        console.error('Error making predictions:', error);
        alert('Error during prediction: ' + error.message);
        
        // Reset button
        buttonText.textContent = '🔮 Make Predictions on Test Data';
        spinner.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function displayPredictionResults() {
    const { passengerIds, probabilities, binaryPredictions } = window.predictions;
    
    const tableBody = document.getElementById('predictionBody');
    tableBody.innerHTML = '';
    
    // Show first 20 predictions
    const displayCount = Math.min(20, passengerIds.length);
    
    for (let i = 0; i < displayCount; i++) {
        const row = document.createElement('tr');
        
        // Color code based on prediction
        if (binaryPredictions[i] === 1) {
            row.classList.add('table-success');
        } else {
            row.classList.add('table-light');
        }
        
        row.innerHTML = `
            <td>${passengerIds[i]}</td>
            <td>${probabilities[i].toFixed(4)}</td>
            <td><strong>${binaryPredictions[i] === 1 ? 'Survived' : 'Perished'}</strong></td>
        `;
        
        tableBody.appendChild(row);
    }
    
    // Add note if more rows exist
    if (passengerIds.length > displayCount) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" class="text-center text-muted">
                ... and ${passengerIds.length - displayCount} more predictions
            </td>
        `;
        tableBody.appendChild(row);
    }
}

// ============================================================================
// EXPORT FUNCTIONS
// ============================================================================

function exportSubmission() {
    if (!window.predictions) {
        alert('Please make predictions first');
        return;
    }
    
    const { passengerIds, binaryPredictions } = window.predictions;
    
    // Create CSV content
    let csvContent = 'PassengerId,Survived\n';
    
    for (let i = 0; i < passengerIds.length; i++) {
        csvContent += `${passengerIds[i]},${binaryPredictions[i]}\n`;
    }
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('Submission CSV exported');
}

function exportProbabilities() {
    if (!window.predictions) {
        alert('Please make predictions first');
        return;
    }
    
    const { passengerIds, probabilities, binaryPredictions, threshold } = window.predictions;
    
    // Create CSV content
    let csvContent = 'PassengerId,Probability,Predicted,Threshold\n';
    
    for (let i = 0; i < passengerIds.length; i++) {
        csvContent += `${passengerIds[i]},${probabilities[i].toFixed(6)},${binaryPredictions[i]},${threshold}\n`;
    }
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'probabilities.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('Probabilities CSV exported');
}

async function saveModel() {
    if (!modelTrained) {
        alert('Please train the model first');
        return;
    }
    
    try {
        // Save model using TensorFlow.js
        await model.save('downloads://titanic-survival-model');
        
        console.log('Model saved successfully');
        alert('Model saved as "titanic-survival-model" in your downloads folder.');
    } catch (error) {
        console.error('Error saving model:', error);
        alert('Error saving model: ' + error.message);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Error handling wrapper for async functions
function handleAsyncError(fn) {
    return function(...args) {
        try {
            return fn(...args).catch(error => {
                console.error('Async error:', error);
                alert('Error: ' + error.message);
            });
        } catch (error) {
            console.error('Sync error:', error);
            alert('Error: ' + error.message);
        }
    };
}

// ============================================================================
// INITIALIZATION
// ============================================================================

console.log('Titanic Survival Predictor loaded successfully');
console.log('TensorFlow.js version:', tf.version.tfjs);
