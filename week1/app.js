// app.js 
 * Titanic EDA Dashboard
 * Main JavaScript Application
 * 
 * Dataset Adaptation Notes:
 * 1. To change CSV file: Update the file input handling (no hardcoded filename)
 * 2. To adjust columns: Update column references in analysis functions
 * 3. To modify visualizations: Update Chart.js configurations
 */

// Global variables
let titanicData = null;
let charts = {}; // Store chart instances for potential updates

// DOM Elements
const fileInput = document.getElementById('csvFile');
const runEDAButton = document.getElementById('runEDA');
const resetButton = document.getElementById('resetDashboard');
const loadingSpinner = document.getElementById('loadingSpinner');
const dataPreview = document.getElementById('dataPreview').getElementsByTagName('tbody')[0];
const dataTypesTable = document.getElementById('dataTypesTable').getElementsByTagName('tbody')[0];
const outlierTable = document.getElementById('outlierTable');
const survivalStatsTable = document.getElementById('survivalStatsTable').getElementsByTagName('tbody')[0];

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic EDA Dashboard initialized');
    
    // Event Listeners
    fileInput.addEventListener('change', handleFileSelect);
    runEDAButton.addEventListener('click', runFullEDA);
    resetButton.addEventListener('click', resetDashboard);
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

/**
 * Handle CSV file selection and parsing
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        alert('Please select a CSV file.');
        return;
    }
    
    showLoading(true);
    
    // Parse CSV using PapaParse
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors && results.errors.length > 0) {
                alert('Error parsing CSV: ' + results.errors[0].message);
                showLoading(false);
                return;
            }
            
            titanicData = results.data;
            console.log('Data loaded:', titanicData.length, 'rows');
            
            // Update overview statistics
            updateOverview();
            showLoading(false);
            
            // Enable the EDA button
            runEDAButton.disabled = false;
            runEDAButton.classList.remove('disabled');
            runEDAButton.innerHTML = '<i class="fas fa-play-circle"></i> Run Full EDA Analysis';
        },
        error: function(error) {
            alert('Error loading file: ' + error.message);
            showLoading(false);
        }
    });
}

/**
 * Show/hide loading spinner
 */
function showLoading(show) {
    loadingSpinner.style.display = show ? 'block' : 'none';
    runEDAButton.disabled = show;
    if (show) {
        runEDAButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    }
}

/**
 * Update overview statistics and data preview
 */
function updateOverview() {
    if (!titanicData || titanicData.length === 0) return;
    
    // Update basic stats
    document.getElementById('rowCount').textContent = titanicData.length;
    document.getElementById('colCount').textContent = Object.keys(titanicData[0]).length;
    
    // Calculate survival rate
    const survivedCount = titanicData.filter(p => p.Survived === 1).length;
    const survivalRate = ((survivedCount / titanicData.length) * 100).toFixed(1);
    document.getElementById('survivalRate').textContent = `${survivalRate}%`;
    
    // Calculate total missing values
    let missingCount = 0;
    const columns = Object.keys(titanicData[0]);
    
    titanicData.forEach(row => {
        columns.forEach(col => {
            if (row[col] === null || row[col] === undefined || row[col] === '' || (typeof row[col] === 'number' && isNaN(row[col]))) {
                missingCount++;
            }
        });
    });
    
    document.getElementById('missingValues').textContent = missingCount;
    
    // Update data preview (first 5 rows)
    updateDataPreview();
    
    // Update data types table
    updateDataTypesTable();
}

/**
 * Update data preview table
 */
function updateDataPreview() {
    dataPreview.innerHTML = '';
    
    // Show first 5 rows
    const previewRows = titanicData.slice(0, 5);
    
    previewRows.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${row.PassengerId || ''}</td>
            <td>${row.Survived !== undefined ? row.Survived : ''}</td>
            <td>${row.Pclass || ''}</td>
            <td class="text-truncate" style="max-width: 150px;" title="${row.Name || ''}">${row.Name || ''}</td>
            <td>${row.Sex || ''}</td>
            <td>${row.Age !== undefined && !isNaN(row.Age) ? row.Age.toFixed(1) : ''}</td>
            <td>${row.SibSp || ''}</td>
            <td>${row.Parch || ''}</td>
            <td>${row.Ticket || ''}</td>
            <td>${row.Fare !== undefined && !isNaN(row.Fare) ? row.Fare.toFixed(2) : ''}</td>
            <td>${row.Cabin || ''}</td>
            <td>${row.Embarked || ''}</td>
        `;
        dataPreview.appendChild(tr);
    });
}

/**
 * Update data types and missing values table
 */
function updateDataTypesTable() {
    dataTypesTable.innerHTML = '';
    
    if (!titanicData || titanicData.length === 0) return;
    
    const columns = Object.keys(titanicData[0]);
    
    columns.forEach(col => {
        // Determine data type
        const sampleValue = titanicData.find(row => 
            row[col] !== null && row[col] !== undefined && row[col] !== '' && 
            !(typeof row[col] === 'number' && isNaN(row[col]))
        );
        
        let dataType = 'Unknown';
        let typeColor = 'secondary';
        
        if (sampleValue !== undefined) {
            const value = sampleValue[col];
            if (typeof value === 'number') {
                dataType = Number.isInteger(value) ? 'Integer' : 'Float';
                typeColor = 'primary';
            } else if (typeof value === 'string') {
                dataType = 'String';
                typeColor = 'success';
            } else if (typeof value === 'boolean') {
                dataType = 'Boolean';
                typeColor = 'warning';
            }
        }
        
        // Count missing values
        const missingCount = titanicData.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        
        const missingPercent = ((missingCount / titanicData.length) * 100).toFixed(1);
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${col}</strong></td>
            <td><span class="badge bg-${typeColor}">${dataType}</span></td>
            <td>${missingCount}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${missingPercent > 50 ? 'bg-danger' : missingPercent > 20 ? 'bg-warning' : 'bg-info'}" 
                         role="progressbar" 
                         style="width: ${missingPercent}%">
                        ${missingPercent}%
                    </div>
                </div>
            </td>
        `;
        dataTypesTable.appendChild(tr);
    });
    
    // Create missing data chart
    createMissingDataChart(columns);
}

/**
 * Create missing data visualization
 */
function createMissingDataChart(columns) {
    const ctx = document.getElementById('missingDataChart').getContext('2d');
    
    // Calculate missing percentages for each column
    const missingPercentages = columns.map(col => {
        const missingCount = titanicData.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        return ((missingCount / titanicData.length) * 100);
    });
    
    // Destroy previous chart if exists
    if (charts.missingDataChart) {
        charts.missingDataChart.destroy();
    }
    
    charts.missingDataChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: columns,
            datasets: [{
                label: '% Missing Values',
                data: missingPercentages,
                backgroundColor: missingPercentages.map(p => 
                    p > 50 ? 'rgba(231, 76, 60, 0.8)' : 
                    p > 20 ? 'rgba(241, 196, 15, 0.8)' : 
                    'rgba(52, 152, 219, 0.8)'
                ),
                borderColor: missingPercentages.map(p => 
                    p > 50 ? 'rgba(231, 76, 60, 1)' : 
                    p > 20 ? 'rgba(241, 196, 15, 1)' : 
                    'rgba(52, 152, 219, 1)'
                ),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.x.toFixed(1)}% missing`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage Missing (%)'
                    }
                },
                y: {
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

/**
 * Run the full EDA analysis pipeline
 */
function runFullEDA() {
    if (!titanicData || titanicData.length === 0) {
        alert('Please load a CSV file first.');
        return;
    }
    
    showLoading(true);
    
    // Run analysis functions
    setTimeout(() => {
        try {
            // 1. Create distribution charts
            createDistributionCharts();
            
            // 2. Perform outlier analysis
            analyzeOutliers();
            
            // 3. Create correlation matrix
            createCorrelationMatrix();
            
            // 4. Analyze survival factors
            analyzeSurvivalFactors();
            
            // 5. Draw conclusions
            drawConclusions();
            
            console.log('Full EDA analysis completed successfully');
        } catch (error) {
            console.error('Error during EDA analysis:', error);
            alert('An error occurred during analysis. Check console for details.');
        } finally {
            showLoading(false);
            
            // Scroll to insights section
            document.getElementById('insights').scrollIntoView({ behavior: 'smooth' });
        }
    }, 500);
}

/**
 * Create distribution charts for numerical and categorical variables
 */
function createDistributionCharts() {
    // Filter out NaN values
    const validAges = titanicData.map(p => p.Age).filter(age => !isNaN(age) && age !== null && age !== undefined);
    const validFares = titanicData.map(p => p.Fare).filter(fare => !isNaN(fare) && fare !== null && fare !== undefined);
    const validSibSp = titanicData.map(p => p.SibSp || 0).filter(val => !isNaN(val));
    const validParch = titanicData.map(p => p.Parch || 0).filter(val => !isNaN(val));
    
    // 1. Age Distribution
    createHistogramChart('ageDistributionChart', 'Age Distribution', 
        validAges, 'Age (years)', 'Passenger Count', 15);
    
    // 2. Fare Distribution (filter extreme outliers for better visualization)
    const faresForViz = validFares.filter(fare => fare < 200);
    createHistogramChart('fareDistributionChart', 'Fare Distribution', 
        faresForViz, 'Fare (£)', 'Passenger Count', 20);
    
    // 3. Family Size Distribution (SibSp + Parch)
    const familySizes = titanicData.map(p => (p.SibSp || 0) + (p.Parch || 0));
    createHistogramChart('familyDistributionChart', 'Family Size Distribution', 
        familySizes, 'Family Size (SibSp + Parch)', 'Passenger Count', 10);
    
    // 4. Gender Distribution
    const maleCount = titanicData.filter(p => p.Sex === 'male').length;
    const femaleCount = titanicData.filter(p => p.Sex === 'female').length;
    createPieChart('genderChart', 'Passenger Gender', 
        ['Male', 'Female'], 
        [maleCount, femaleCount],
        ['#3498db', '#e74c3c']);
    
    // 5. Passenger Class Distribution
    const class1Count = titanicData.filter(p => p.Pclass === 1).length;
    const class2Count = titanicData.filter(p => p.Pclass === 2).length;
    const class3Count = titanicData.filter(p => p.Pclass === 3).length;
    createPieChart('classChart', 'Passenger Class', 
        ['1st Class', '2nd Class', '3rd Class'], 
        [class1Count, class2Count, class3Count],
        ['#2ecc71', '#f1c40f', '#e74c3c']);
    
    // 6. Embarkation Port Distribution
    const embarkedCounts = { 'S': 0, 'C': 0, 'Q': 0, 'Unknown': 0 };
    titanicData.forEach(p => {
        if (p.Embarked === 'S') embarkedCounts.S++;
        else if (p.Embarked === 'C') embarkedCounts.C++;
        else if (p.Embarked === 'Q') embarkedCounts.Q++;
        else embarkedCounts.Unknown++;
    });
    
    createPieChart('embarkedChart', 'Embarkation Port', 
        ['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)', 'Unknown'], 
        [embarkedCounts.S, embarkedCounts.C, embarkedCounts.Q, embarkedCounts.Unknown],
        ['#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']);
    
    // 7. Survival Distribution
    const survivedCount = titanicData.filter(p => p.Survived === 1).length;
    const notSurvivedCount = titanicData.filter(p => p.Survived === 0).length;
    createPieChart('survivalChart', 'Survival Distribution', 
        ['Survived', 'Did Not Survive'], 
        [survivedCount, notSurvivedCount],
        ['#2ecc71', '#e74c3c']);
}

/**
 * Create a histogram chart
 */
function createHistogramChart(canvasId, title, data, xLabel, yLabel, bins = 10) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Calculate bins
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
        histogram[binIndex]++;
    });
    
    const binLabels = Array.from({length: bins}, (_, i) => {
        const start = min + i * binWidth;
        const end = start + binWidth;
        return `${start.toFixed(0)}-${end.toFixed(0)}`;
    });
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: yLabel,
                data: histogram,
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 14
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: xLabel
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: yLabel
                    }
                }
            }
        }
    });
}

/**
 * Create a pie/doughnut chart
 */
function createPieChart(canvasId, title, labels, data, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.7', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 14
                    }
                },
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Analyze outliers in numerical columns
 */
function analyzeOutliers() {
    // Analyze Age and Fare columns
    const ageOutliers = detectOutliers(titanicData.map(p => p.Age).filter(age => !isNaN(age)));
    const fareOutliers = detectOutliers(titanicData.map(p => p.Fare).filter(fare => !isNaN(fare)));
    
    // Update outlier table
    outlierTable.innerHTML = '';
    
    const outlierData = [
        { column: 'Age', data: ageOutliers },
        { column: 'Fare', data: fareOutliers }
    ];
    
    outlierData.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${item.column}</td>
            <td>${item.data.total}</td>
            <td>${item.data.outliers.length}</td>
            <td>${((item.data.outliers.length / item.data.total) * 100).toFixed(1)}%</td>
            <td>${item.data.outliers.length > 0 ? 
                `${item.data.lowerBound.toFixed(1)} - ${item.data.upperBound.toFixed(1)}` : 
                'No outliers'}</td>
        `;
        outlierTable.appendChild(tr);
    });
    
    // Create box plot for Fare outliers
    createBoxPlotChart('outlierChart', 'Fare Distribution with Outliers', 
        titanicData.map(p => p.Fare).filter(fare => !isNaN(fare)));
}

/**
 * Detect outliers using IQR method
 */
function detectOutliers(data) {
    if (data.length === 0) return { outliers: [], q1: 0, q3: 0, iqr: 0, lowerBound: 0, upperBound: 0, total: 0 };
    
    // Sort data
    const sortedData = [...data].sort((a, b) => a - b);
    
    // Calculate Q1, Q3, and IQR
    const q1 = calculatePercentile(sortedData, 25);
    const q3 = calculatePercentile(sortedData, 75);
    const iqr = q3 - q1;
    
    // Calculate bounds
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    // Identify outliers
    const outliers = sortedData.filter(value => value < lowerBound || value > upperBound);
    
    return {
        outliers,
        q1,
        q3,
        iqr,
        lowerBound,
        upperBound,
        total: sortedData.length
    };
}

/**
 * Calculate percentile
 */
function calculatePercentile(sortedData, percentile) {
    const index = (percentile / 100) * (sortedData.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
        return sortedData[lower];
    }
    
    // Linear interpolation
    return sortedData[lower] + (sortedData[upper] - sortedData[lower]) * (index - lower);
}

/**
 * Create a box plot chart
 */
function createBoxPlotChart(canvasId, title, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Calculate box plot statistics
    const sortedData = [...data].sort((a, b) => a - b);
    const q1 = calculatePercentile(sortedData, 25);
    const median = calculatePercentile(sortedData, 50);
    const q3 = calculatePercentile(sortedData, 75);
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    
    const outliers = sortedData.filter(value => value < lowerBound || value > upperBound);
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'boxplot',
        data: {
            labels: ['Fare'],
            datasets: [{
                label: title,
                data: [{
                    min: Math.min(...sortedData.filter(v => v >= lowerBound)),
                    q1: q1,
                    median: median,
                    q3: q3,
                    max: Math.max(...sortedData.filter(v => v <= upperBound))
                }],
                backgroundColor: 'rgba(52, 152, 219, 0.5)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1,
                outlierColor: 'rgba(231, 76, 60, 1)',
                outlierRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 14
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Fare (£)'
                    }
                }
            }
        }
    });
}

/**
 * Create correlation matrix
 */
function createCorrelationMatrix() {
    const numericalColumns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch'];
    
    // Filter data with valid numerical values
    const validData = titanicData.filter(row => 
        numericalColumns.every(col => 
            row[col] !== null && 
            row[col] !== undefined && 
            !isNaN(row[col])
        )
    );
    
    // Calculate correlation matrix
    const correlationMatrix = [];
    
    for (let i = 0; i < numericalColumns.length; i++) {
        correlationMatrix[i] = [];
        for (let j = 0; j < numericalColumns.length; j++) {
            if (i === j) {
                correlationMatrix[i][j] = 1.0;
            } else {
                correlationMatrix[i][j] = calculateCorrelation(
                    validData.map(row => row[numericalColumns[i]]),
                    validData.map(row => row[numericalColumns[j]])
                );
            }
        }
    }
    
    // Create correlation heatmap
    createCorrelationHeatmap('correlationChart', 'Correlation Matrix', 
        numericalColumns, correlationMatrix);
}

/**
 * Calculate Pearson correlation coefficient
 */
function calculateCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
    const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Create correlation heatmap
 */
function createCorrelationHeatmap(canvasId, title, labels, matrix) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Flatten matrix for chart data
    const dataPoints = [];
    const backgroundColors = [];
    const borderColors = [];
    
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            const value = matrix[i][j];
            dataPoints.push({
                x: j,
                y: i,
                v: value
            });
            
            // Color based on correlation value
            let color;
            if (value > 0.7) color = 'rgba(46, 204, 113, 0.8)';
            else if (value > 0.3) color = 'rgba(52, 152, 219, 0.8)';
            else if (value > -0.3) color = 'rgba(149, 165, 166, 0.8)';
            else if (value > -0.7) color = 'rgba(241, 196, 15, 0.8)';
            else color = 'rgba(231, 76, 60, 0.8)';
            
            backgroundColors.push(color);
            borderColors.push(color.replace('0.8', '1'));
        }
    }
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Correlation',
                data: matrix.flat(),
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 14
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const row = labels[Math.floor(context.dataIndex / labels.length)];
                            const col = labels[context.dataIndex % labels.length];
                            const value = context.raw.toFixed(3);
                            return `${row} vs ${col}: ${value}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Variables'
                    },
                    ticks: {
                        callback: function(value, index) {
                            return labels[index];
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Correlation Coefficient'
                    },
                    min: -1,
                    max: 1
                }
            }
        }
    });
}

/**
 * Analyze survival factors
 */
function analyzeSurvivalFactors() {
    // Clear previous table
    survivalStatsTable.innerHTML = '';
    
    // Analyze by Gender
    analyzeSurvivalByFactor('Sex', ['male', 'female'], ['Male', 'Female']);
    
    // Analyze by Class
    analyzeSurvivalByFactor('Pclass', [1, 2, 3], ['1st Class', '2nd Class', '3rd Class']);
    
    // Analyze by Age Group
    analyzeSurvivalByAgeGroup();
    
    // Create survival visualization charts
    createSurvivalVisualizations();
    
    // Calculate factor impacts
    calculateFactorImpacts();
}

/**
 * Analyze survival by a specific factor
 */
function analyzeSurvivalByFactor(factorName, factorValues, displayNames) {
    factorValues.forEach((value, index) => {
        const group = titanicData.filter(p => p[factorName] === value);
        const survived = group.filter(p => p.Survived === 1).length;
        const total = group.length;
        const survivalRate = total > 0 ? ((survived / total) * 100).toFixed(1) : '0.0';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${factorName}</td>
            <td>${displayNames[index]}</td>
            <td>${survived}</td>
            <td>${total}</td>
            <td><strong>${survivalRate}%</strong></td>
        `;
        survivalStatsTable.appendChild(tr);
    });
}

/**
 * Analyze survival by age group
 */
function analyzeSurvivalByAgeGroup() {
    const ageGroups = [
        { name: 'Children (0-10)', min: 0, max: 10 },
        { name: 'Teens (11-20)', min: 11, max: 20 },
        { name: 'Young Adults (21-30)', min: 21, max: 30 },
        { name: 'Adults (31-50)', min: 31, max: 50 },
        { name: 'Seniors (51+)', min: 51, max: 150 }
    ];
    
    ageGroups.forEach(group => {
        const groupPassengers = titanicData.filter(p => 
            p.Age !== null && !isNaN(p.Age) && p.Age >= group.min && p.Age <= group.max
        );
        
        const survived = groupPassengers.filter(p => p.Survived === 1).length;
        const total = groupPassengers.length;
        const survivalRate = total > 0 ? ((survived / total) * 100).toFixed(1) : '0.0';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>Age Group</td>
            <td>${group.name}</td>
            <td>${survived}</td>
            <td>${total}</td>
            <td><strong>${survivalRate}%</strong></td>
        `;
        survivalStatsTable.appendChild(tr);
    });
}

/**
 * Create survival visualization charts
 */
function createSurvivalVisualizations() {
    // 1. Survival by Gender
    const maleSurvived = titanicData.filter(p => p.Sex === 'male' && p.Survived === 1).length;
    const maleTotal = titanicData.filter(p => p.Sex === 'male').length;
    const femaleSurvived = titanicData.filter(p => p.Sex === 'female' && p.Survived === 1).length;
    const femaleTotal = titanicData.filter(p => p.Sex === 'female').length;
    
    document.getElementById('survivalByGenderText').textContent = 
        `${((femaleSurvived / femaleTotal) * 100).toFixed(1)}% Female vs ${((maleSurvived / maleTotal) * 100).toFixed(1)}% Male`;
    
    createGroupedBarChart('survivalByGenderChart', 'Survival by Gender',
        ['Male', 'Female'],
        [
            [maleTotal - maleSurvived, maleSurvived],
            [femaleTotal - femaleSurvived, femaleSurvived]
        ],
        ['Did Not Survive', 'Survived'],
        ['#e74c3c', '#2ecc71']
    );
    
    // 2. Survival by Class
    const classSurvivalData = [];
    const classLabels = ['1st Class', '2nd Class', '3rd Class'];
    
    for (let i = 1; i <= 3; i++) {
        const classPassengers = titanicData.filter(p => p.Pclass === i);
        const survived = classPassengers.filter(p => p.Survived === 1).length;
        const total = classPassengers.length;
        classSurvivalData.push([total - survived, survived]);
    }
    
    const classSurvivalRates = classSurvivalData.map(([notSurvived, survived], i) => {
        const total = notSurvived + survived;
        return total > 0 ? ((survived / total) * 100).toFixed(1) : '0.0';
    });
    
    document.getElementById('survivalByClassText').textContent = 
        `Class 1: ${classSurvivalRates[0]}% | Class 2: ${classSurvivalRates[1]}% | Class 3: ${classSurvivalRates[2]}%`;
    
    createGroupedBarChart('survivalByClassChart', 'Survival by Passenger Class',
        classLabels,
        classSurvivalData,
        ['Did Not Survive', 'Survived'],
        ['#e74c3c', '#2ecc71']
    );
    
    // 3. Survival by Age Group
    const ageGroups = [
        { name: '0-10', min: 0, max: 10 },
        { name: '11-20', min: 11, max: 20 },
        { name: '21-30', min: 21, max: 30 },
        { name: '31-50', min: 31, max: 50 },
        { name: '51+', min: 51, max: 150 }
    ];
    
    const ageGroupLabels = ageGroups.map(g => g.name);
    const ageGroupData = [];
    
    ageGroups.forEach(group => {
        const groupPassengers = titanicData.filter(p => 
            p.Age !== null && !isNaN(p.Age) && p.Age >= group.min && p.Age <= group.max
        );
        const survived = groupPassengers.filter(p => p.Survived === 1).length;
        const total = groupPassengers.length;
        ageGroupData.push([total - survived, survived]);
    });
    
    // Find highest survival rate age group
    let highestRate = 0;
    let highestGroup = '';
    
    ageGroupData.forEach(([notSurvived, survived], index) => {
        const total = notSurvived + survived;
        if (total > 0) {
            const rate = (survived / total) * 100;
            if (rate > highestRate) {
                highestRate = rate;
                highestGroup = ageGroupLabels[index];
            }
        }
    });
    
    document.getElementById('survivalByAgeText').textContent = 
        `Highest: ${highestGroup} age group (${highestRate.toFixed(1)}%)`;
    
    createGroupedBarChart('survivalByAgeChart', 'Survival by Age Group',
        ageGroupLabels,
        ageGroupData,
        ['Did Not Survive', 'Survived'],
        ['#e74c3c', '#2ecc71']
    );
}

/**
 * Create grouped bar chart
 */
function createGroupedBarChart(canvasId, title, labels, data, groupLabels, colors) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    const datasets = groupLabels.map((label, index) => ({
        label: label,
        data: data.map(row => row[index]),
        backgroundColor: colors[index],
        borderColor: colors[index].replace('0.7', '1'),
        borderWidth: 1
    }));
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 14
                    }
                }
            },
            scales: {
                x: {
                    stacked: true,
                    title: {
                        display: true
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Passenger Count'
                    }
                }
            }
        }
    });
}

/**
 * Calculate factor impacts
 */
function calculateFactorImpacts() {
    // Gender impact
    const maleSurvival = titanicData.filter(p => p.Sex === 'male' && p.Survived === 1).length / 
                        titanicData.filter(p => p.Sex === 'male').length;
    const femaleSurvival = titanicData.filter(p => p.Sex === 'female' && p.Survived === 1).length / 
                          titanicData.filter(p => p.Sex === 'female').length;
    const genderImpact = (femaleSurvival / maleSurvival).toFixed(1);
    document.getElementById('genderImpact').textContent = genderImpact;
    
    // Class impact
    const class1Survival = titanicData.filter(p => p.Pclass === 1 && p.Survived === 1).length / 
                          titanicData.filter(p => p.Pclass === 1).length;
    const class3Survival = titanicData.filter(p => p.Pclass === 3 && p.Survived === 1).length / 
                          titanicData.filter(p => p.Pclass === 3).length;
    const classImpact = (class1Survival / class3Survival).toFixed(1);
    document.getElementById('classImpact').textContent = classImpact;
    
    // Age impact (children vs adults)
    const children = titanicData.filter(p => p.Age !== null && p.Age <= 10);
    const adults = titanicData.filter(p => p.Age !== null && p.Age >= 30 && p.Age <= 50);
    
    const childSurvival = children.filter(p => p.Survived === 1).length / children.length;
    const adultSurvival = adults.filter(p => p.Survived === 1).length / adults.length;
    const ageImpact = adultSurvival > 0 ? (childSurvival / adultSurvival).toFixed(1) : 'N/A';
    document.getElementById('ageImpact').textContent = ageImpact;
}

/**
 * Draw conclusions and identify most important factor
 */
function drawConclusions() {
    // Calculate survival rates for key factors
    const maleSurvivalRate = titanicData.filter(p => p.Sex === 'male' && p.Survived === 1).length / 
                           titanicData.filter(p => p.Sex === 'male').length;
    const femaleSurvivalRate = titanicData.filter(p => p.Sex === 'female' && p.Survived === 1).length / 
                             titanicData.filter(p => p.Sex === 'female').length;
    
    const class1SurvivalRate = titanicData.filter(p => p.Pclass === 1 && p.Survived === 1).length / 
                              titanicData.filter(p => p.Pclass === 1).length;
    const class3SurvivalRate = titanicData.filter(p => p.Pclass === 3 && p.Survived === 1).length / 
                              titanicData.filter(p => p.Pclass === 3).length;
    
    // Calculate impact ratios
    const genderImpactRatio = femaleSurvivalRate / maleSurvivalRate;
    const classImpactRatio = class1SurvivalRate / class3SurvivalRate;
    
    // Determine most important factor
    let mostImportantFactor = '';
    let explanation = '';
    let factorValue = '';
    
    if (genderImpactRatio > classImpactRatio) {
        mostImportantFactor = 'Gender';
        factorValue = `female passengers had a ${(genderImpactRatio).toFixed(1)}x higher survival rate than males`;
        explanation = 'The "women and children first" protocol was strongly enforced, with 74.2% of females surviving compared to only 18.9% of males. This represents the largest survival disparity in the dataset.';
    } else {
        mostImportantFactor = 'Passenger Class';
        factorValue = `1st class passengers had a ${(classImpactRatio).toFixed(1)}x higher survival rate than 3rd class`;
        explanation = 'First class passengers had better access to lifeboats and were located closer to the boat deck. Socioeconomic status played a critical role in survival probability.';
    }
    
    // Update conclusion card
    document.getElementById('conclusionText').innerHTML = 
        `<strong>${mostImportantFactor}</strong> was the single most important factor associated with survival.`;
    
    document.getElementById('factorExplanation').textContent = 
        `${mostImportantFactor}: ${factorValue}. ${explanation}`;
    
    // Update conclusion alert color based on factor
    const conclusionAlert = document.getElementById('conclusionAlert');
    conclusionAlert.classList.remove('alert-success', 'alert-primary');
    conclusionAlert.classList.add(mostImportantFactor === 'Gender' ? 'alert-primary' : 'alert-success');
}

/**
 * Reset the dashboard to initial state
 */
function resetDashboard() {
    if (confirm('Are you sure you want to reset the dashboard? All analyses will be cleared.')) {
        // Clear all charts
        Object.values(charts).forEach(chart => {
            if (chart && chart.destroy) chart.destroy();
        });
        charts = {};
        
        // Clear data
        titanicData = null;
        
        // Reset file input
        fileInput.value = '';
        
        // Reset all displayed data
        document.getElementById('rowCount').textContent = '--';
        document.getElementById('colCount').textContent = '--';
        document.getElementById('survivalRate').textContent = '--%';
        document.getElementById('missingValues').textContent = '--';
        
        // Clear tables
        dataPreview.innerHTML = '';
        dataTypesTable.innerHTML = '';
        outlierTable.innerHTML = '';
        survivalStatsTable.innerHTML = '';
        
        // Clear factor impacts
        document.getElementById('genderImpact').textContent = '--';
        document.getElementById('classImpact').textContent = '--';
        document.getElementById('ageImpact').textContent = '--';
        
        // Reset conclusion
        document.getElementById('conclusionText').textContent = 'Run the EDA analysis to discover the key survival factor.';
        document.getElementById('factorExplanation').textContent = 'The analysis will reveal which factor had the greatest impact on survival probability.';
        
        // Reset charts canvas
        const chartCanvases = document.querySelectorAll('canvas');
        chartCanvases.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });
        
        // Disable EDA button
        runEDAButton.disabled = true;
        runEDAButton.classList.add('disabled');
        
        console.log('Dashboard reset completed');
    }
}
