// app.js
/**
 * Titanic EDA Dashboard
 * Client-Side Exploratory Data Analysis Application
 * Uses: PapaParse (CSV parsing), Chart.js (visualizations), Bootstrap 5 (UI)
 */

// Global variables to store data and chart instances
let titanicData = [];
let charts = {};
let columnStats = {};

// DOM Content Loaded Event Listener
document.addEventListener('DOMContentLoaded', function() {
    // Initialize event listeners
    document.getElementById('runEDA').addEventListener('click', loadAndAnalyzeData);
    document.getElementById('csvFile').addEventListener('change', handleFileSelect);
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        document.getElementById('runButtonText').textContent = `Analyze ${file.name}`;
    }
}

// Main function to load and analyze data
function loadAndAnalyzeData() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first. Use Titanic-Dataset.csv from Kaggle.');
        return;
    }
    
    // Show loading state
    const runButton = document.getElementById('runEDA');
    const buttonText = document.getElementById('runButtonText');
    const spinner = document.getElementById('loadingSpinner');
    
    buttonText.textContent = 'Analyzing...';
    spinner.style.display = 'inline-block';
    runButton.disabled = true;
    
    // Parse CSV file using PapaParse
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
                alert('Error parsing CSV: ' + results.errors[0].message);
                resetButtonState();
                return;
            }
            
            titanicData = results.data;
            console.log(`Loaded ${titanicData.length} records with ${Object.keys(titanicData[0]).length} columns`);
            
            // Perform complete EDA pipeline
            performEDA();
            
            // Reset button state
            resetButtonState();
        },
        error: function(error) {
            alert('Error loading file: ' + error.message);
            resetButtonState();
        }
    });
    
    function resetButtonState() {
        buttonText.textContent = '🔄 Re-run EDA Analysis';
        spinner.style.display = 'none';
        runButton.disabled = false;
    }
}

// Main EDA pipeline
function performEDA() {
    // Show all sections
    document.querySelectorAll('.card').forEach(card => {
        card.style.display = 'block';
    });
    
    // 1. Data Overview
    displayDataOverview();
    
    // 2. Data Types & Missing Values
    analyzeDataTypesAndMissing();
    
    // 3. Distribution Analysis
    createDistributionCharts();
    
    // 4. Outlier Detection
    detectOutliers();
    
    // 5. Correlation Analysis
    calculateCorrelations();
    
    // 6. Survival Analysis
    analyzeSurvivalFactors();
    
    // 7. Key Insights & Conclusion
    generateInsightsAndConclusion();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 1. Display Data Overview
function displayDataOverview() {
    const rowCount = titanicData.length;
    const colCount = Object.keys(titanicData[0]).length;
    
    // Update overview stats
    document.getElementById('rowCount').textContent = rowCount.toLocaleString();
    document.getElementById('colCount').textContent = colCount;
    
    // Calculate survival rate
    const survivedCount = titanicData.filter(row => row.Survived === 1).length;
    const survivalRate = ((survivedCount / rowCount) * 100).toFixed(1);
    document.getElementById('survivalRate').textContent = `${survivalRate}%`;
    
    // Calculate complete cases
    const completeCases = titanicData.filter(row => {
        return Object.values(row).every(value => value !== null && value !== undefined && value !== '');
    }).length;
    const completeCasesPct = ((completeCases / rowCount) * 100).toFixed(1);
    document.getElementById('completeCases').textContent = `${completeCasesPct}%`;
    
    // Display data preview
    displayDataPreview();
    
    // Show overview section
    document.getElementById('dataOverview').style.display = 'block';
}

// Display data preview table
function displayDataPreview() {
    const previewHeader = document.getElementById('previewHeader');
    const previewBody = document.getElementById('previewBody');
    
    // Clear existing content
    previewHeader.innerHTML = '';
    previewBody.innerHTML = '';
    
    // Get column names from first row
    const columns = Object.keys(titanicData[0]);
    
    // Create header row
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        th.scope = 'col';
        headerRow.appendChild(th);
    });
    previewHeader.appendChild(headerRow);
    
    // Create data rows (first 10)
    const previewRows = titanicData.slice(0, 10);
    previewRows.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            let value = row[col];
            
            // Handle null/undefined values
            if (value === null || value === undefined || value === '') {
                value = '<span class="text-muted">-</span>';
                td.innerHTML = value;
            } else if (typeof value === 'number') {
                td.textContent = Number.isInteger(value) ? value : value.toFixed(2);
            } else {
                td.textContent = value;
            }
            
            tr.appendChild(td);
        });
        previewBody.appendChild(tr);
    });
}

// 2. Analyze Data Types and Missing Values
function analyzeDataTypesAndMissing() {
    const columnInfoBody = document.getElementById('columnInfo');
    columnInfoBody.innerHTML = '';
    
    const columns = Object.keys(titanicData[0]);
    let totalMissing = 0;
    let totalCells = titanicData.length * columns.length;
    
    columns.forEach(col => {
        const nonMissingCount = titanicData.filter(row => {
            const value = row[col];
            return value !== null && value !== undefined && value !== '';
        }).length;
        
        const missingCount = titanicData.length - nonMissingCount;
        const missingPct = ((missingCount / titanicData.length) * 100).toFixed(1);
        
        totalMissing += missingCount;
        
        // Determine data type
        const sampleValue = titanicData.find(row => row[col] !== null && row[col] !== undefined && row[col] !== '');
        let dataType = sampleValue ? typeof sampleValue[col] : 'unknown';
        
        // Special handling for numeric columns that might be categorical
        if (dataType === 'number' && ['Pclass', 'SibSp', 'Parch', 'Survived'].includes(col)) {
            dataType = 'integer (categorical)';
        } else if (dataType === 'number') {
            dataType = 'float';
        } else if (dataType === 'string') {
            dataType = 'text';
        }
        
        // Store column stats for later use
        columnStats[col] = {
            missingCount,
            missingPct,
            dataType,
            nonMissingCount
        };
        
        // Add to table
        const tr = document.createElement('tr');
        tr.innerHTML = `
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
        `;
        columnInfoBody.appendChild(tr);
    });
    
    // Update missing summary
    const missingSummary = document.getElementById('missingSummary');
    const totalMissingPct = ((totalMissing / totalCells) * 100).toFixed(1);
    missingSummary.textContent = `${totalMissing} missing values (${totalMissingPct}% of all data). Age (${columnStats['Age']?.missingPct || 0}%) and Cabin (${columnStats['Cabin']?.missingPct || 0}%) have the most missing values.`;
    
    // Create missing values chart
    createMissingValuesChart();
}

// Create missing values bar chart
function createMissingValuesChart() {
    const ctx = document.getElementById('missingValuesChart').getContext('2d');
    
    // Sort columns by missing percentage
    const columns = Object.keys(columnStats).sort((a, b) => 
        columnStats[b].missingPct - columnStats[a].missingPct
    );
    
    const labels = columns;
    const missingData = columns.map(col => parseFloat(columnStats[col].missingPct));
    
    // Destroy previous chart if exists
    if (charts.missingValues) {
        charts.missingValues.destroy();
    }
    
    charts.missingValues = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Missing Values %',
                data: missingData,
                backgroundColor: missingData.map(pct => 
                    pct > 50 ? 'rgba(231, 76, 60, 0.7)' : 
                    pct > 20 ? 'rgba(241, 196, 15, 0.7)' : 
                    'rgba(52, 152, 219, 0.7)'
                ),
                borderColor: missingData.map(pct => 
                    pct > 50 ? 'rgb(231, 76, 60)' : 
                    pct > 20 ? 'rgb(241, 196, 15)' : 
                    'rgb(52, 152, 219)'
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
                            return `Missing: ${context.raw}%`;
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
                        text: 'Percentage Missing'
                    }
                },
                y: {
                    ticks: {
                        autoSkip: false
                    }
                }
            }
        }
    });
}

// 3. Create Distribution Charts
function createDistributionCharts() {
    createNumericalDistributionCharts();
    createCategoricalDistributionCharts();
}

// Create numerical distribution charts
function createNumericalDistributionCharts() {
    // Age Distribution
    const ageData = titanicData
        .filter(row => row.Age !== null && row.Age !== undefined && !isNaN(row.Age))
        .map(row => row.Age);
    
    createHistogram('ageDistChart', ageData, 'Age', 'Passenger Count', 'Age Distribution');
    
    // Fare Distribution
    const fareData = titanicData
        .filter(row => row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare))
        .map(row => row.Fare);
    
    createHistogram('fareDistChart', fareData, 'Fare', 'Passenger Count', 'Fare Distribution', 20, true);
    
    // SibSp Distribution (as histogram since it's discrete)
    const sibspData = titanicData
        .filter(row => row.SibSp !== null && row.SibSp !== undefined && !isNaN(row.SibSp))
        .map(row => row.SibSp);
    
    createHistogram('sibspDistChart', sibspData, 'Siblings/Spouses', 'Passenger Count', 'SibSp Distribution', 7);
    
    // Parch Distribution (as histogram since it's discrete)
    const parchData = titanicData
        .filter(row => row.Parch !== null && row.Parch !== undefined && !isNaN(row.Parch))
        .map(row => row.Parch);
    
    createHistogram('parchDistChart', parchData, 'Parents/Children', 'Passenger Count', 'Parch Distribution', 7);
}

// Create categorical distribution charts
function createCategoricalDistributionCharts() {
    // Gender Distribution
    const sexCounts = countCategories('Sex');
    createBarChart('sexDistChart', 
        Object.keys(sexCounts), 
        Object.values(sexCounts), 
        'Gender', 
        'Count', 
        'Gender Distribution',
        ['rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
    );
    
    // Passenger Class Distribution
    const pclassCounts = countCategories('Pclass');
    createBarChart('pclassDistChart', 
        Object.keys(pclassCounts).map(key => `Class ${key}`), 
        Object.values(pclassCounts), 
        'Passenger Class', 
        'Count', 
        'Passenger Class Distribution',
        ['rgba(46, 204, 113, 0.7)', 'rgba(52, 152, 219, 0.7)', 'rgba(155, 89, 182, 0.7)']
    );
    
    // Embarked Port Distribution
    const embarkedCounts = countCategories('Embarked');
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
    
    // Survival Distribution
    const survivedCounts = countCategories('Survived');
    createBarChart('survivedDistChart', 
        ['Perished', 'Survived'], 
        [survivedCounts[0] || 0, survivedCounts[1] || 0], 
        'Outcome', 
        'Count', 
        'Survival Distribution',
        ['rgba(231, 76, 60, 0.7)', 'rgba(46, 204, 113, 0.7)']
    );
}

// Helper function to count categories
function countCategories(columnName) {
    const counts = {};
    titanicData.forEach(row => {
        const value = row[columnName];
        if (value !== null && value !== undefined && value !== '') {
            counts[value] = (counts[value] || 0) + 1;
        }
    });
    return counts;
}

// Helper function to create histogram
function createHistogram(canvasId, data, xLabel, yLabel, title, bins = 15, logScale = false) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Calculate bin ranges
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binWidth = (max - min) / bins;
    
    // Create bins
    const binCounts = new Array(bins).fill(0);
    data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
        binCounts[binIndex]++;
    });
    
    // Create bin labels
    const binLabels = [];
    for (let i = 0; i < bins; i++) {
        const binStart = min + (i * binWidth);
        const binEnd = min + ((i + 1) * binWidth);
        binLabels.push(`${binStart.toFixed(1)}-${binEnd.toFixed(1)}`);
    }
    
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
                title: {
                    display: true,
                    text: title
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
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        callback: function(value, index) {
                            // Show fewer labels for clarity
                            return index % Math.ceil(bins / 10) === 0 ? this.getLabelForValue(value) : '';
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: yLabel
                    },
                    beginAtZero: true,
                    ...(logScale && { type: 'logarithmic' })
                }
            }
        }
    });
}

// Helper function to create bar chart
function createBarChart(canvasId, labels, data, xLabel, yLabel, title, colors = null) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy previous chart if exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    // Generate colors if not provided
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
                title: {
                    display: true,
                    text: title
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
                    title: {
                        display: true,
                        text: yLabel
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// 4. Detect Outliers
function detectOutliers() {
    // Detect outliers for Age
    const ageData = titanicData
        .filter(row => row.Age !== null && row.Age !== undefined && !isNaN(row.Age))
        .map(row => row.Age);
    
    const ageOutliers = detectOutliersIQR(ageData);
    const ageOutliersCount = ageOutliers.length;
    const ageOutliersPct = ((ageOutliersCount / ageData.length) * 100).toFixed(1);
    
    document.getElementById('ageOutliers').textContent = ageOutliersCount;
    document.getElementById('ageOutliersPct').textContent = `${ageOutliersPct}%`;
    
    // Detect outliers for Fare
    const fareData = titanicData
        .filter(row => row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare))
        .map(row => row.Fare);
    
    const fareOutliers = detectOutliersIQR(fareData);
    const fareOutliersCount = fareOutliers.length;
    const fareOutliersPct = ((fareOutliersCount / fareData.length) * 100).toFixed(1);
    
    document.getElementById('fareOutliers').textContent = fareOutliersCount;
    document.getElementById('fareOutliersPct').textContent = `${fareOutliersPct}%`;
    
    // Create box plot for Fare outliers
    createBoxPlot(fareData, fareOutliers);
}

// Detect outliers using IQR method
function detectOutliersIQR(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - (1.5 * iqr);
    const upperBound = q3 + (1.5 * iqr);
    
    return data.filter(value => value < lowerBound || value > upperBound);
}

// Create box plot visualization
function createBoxPlot(data, outliers) {
    const ctx = document.getElementById('outlierChart').getContext('2d');
    
    // Calculate statistics
    const sorted = [...data].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const min = Math.min(...data.filter(value => value >= q1 - (1.5 * iqr)));
    const max = Math.max(...data.filter(value => value <= q3 + (1.5 * iqr)));
    
    // Destroy previous chart if exists
    if (charts.outlierChart) {
        charts.outlierChart.destroy();
    }
    
    // Create box plot data
    const boxPlotData = {
        min: min,
        q1: q1,
        median: median,
        q3: q3,
        max: max,
        outliers: outliers
    };
    
    charts.outlierChart = new Chart(ctx, {
        type: 'boxplot',
        data: {
            labels: ['Fare'],
            datasets: [{
                label: 'Fare Distribution',
                data: [boxPlotData],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgb(52, 152, 219)',
                borderWidth: 2,
                outlierBackgroundColor: 'rgba(231, 76, 60, 0.7)',
                outlierBorderColor: 'rgb(231, 76, 60)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Fare Distribution with Outliers (Box Plot)'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Fare Amount'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// 5. Calculate Correlations
function calculateCorrelations() {
    const numericalColumns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Survived'];
    
    // Filter data to only include rows with all numerical values
    const completeData = titanicData.filter(row => 
        numericalColumns.every(col => 
            row[col] !== null && row[col] !== undefined && !isNaN(row[col])
        )
    );
    
    // Calculate correlation matrix
    const correlationMatrix = [];
    for (let i = 0; i < numericalColumns.length; i++) {
        correlationMatrix[i] = [];
        for (let j = 0; j < numericalColumns.length; j++) {
            correlationMatrix[i][j] = calculateCorrelation(
                completeData.map(row => row[numericalColumns[i]]),
                completeData.map(row => row[numericalColumns[j]])
            );
        }
    }
    
    // Create correlation heatmap
    createCorrelationHeatmap(numericalColumns, correlationMatrix);
    
    // Generate correlation insights
    generateCorrelationInsights(numericalColumns, correlationMatrix);
}

// Calculate Pearson correlation coefficient
function calculateCorrelation(x, y) {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0);
    const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
    const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
    
    const numerator = (n * sumXY) - (sumX * sumY);
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
}

// Create correlation heatmap
function createCorrelationHeatmap(labels, matrix) {
    const ctx = document.getElementById('correlationChart').getContext('2d');
    
    // Flatten matrix for chart.js
    const data = [];
    for (let i = 0; i < labels.length; i++) {
        for (let j = 0; j < labels.length; j++) {
            data.push({
                x: labels[j],
                y: labels[i],
                v: matrix[i][j]
            });
        }
    }
    
    // Destroy previous chart if exists
    if (charts.correlationChart) {
        charts.correlationChart.destroy();
    }
    
    charts.correlationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            datasets: [{
                label: 'Correlation',
                data: data,
                backgroundColor: function(context) {
                    const value = context.dataset.data[context.dataIndex].v;
                    const alpha = Math.abs(value);
                    if (value > 0) {
                        return `rgba(46, 204, 113, ${alpha})`;
                    } else if (value < 0) {
                        return `rgba(231, 76, 60, ${alpha})`;
                    } else {
                        return 'rgba(149, 165, 166, 0.2)';
                    }
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Correlation Matrix Heatmap'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.dataset.data[context.dataIndex].v;
                            return `Correlation: ${value.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'category',
                    labels: labels,
                    title: {
                        display: true,
                        text: 'Features'
                    }
                },
                y: {
                    type: 'category',
                    labels: labels,
                    title: {
                        display: true,
                        text: 'Features'
                    }
                },
                color: {
                    type: 'linear',
                    display: false,
                    min: -1,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

// Generate correlation insights
function generateCorrelationInsights(labels, matrix) {
    const insightsList = document.getElementById('correlationInsights');
    insightsList.innerHTML = '';
    
    // Find strongest correlations
    const insights = [];
    
    for (let i = 0; i < labels.length; i++) {
        for (let j = i + 1; j < labels.length; j++) {
            const corr = matrix[i][j];
            if (Math.abs(corr) > 0.3) {
                const strength = Math.abs(corr) > 0.7 ? 'Strong' : 
                                Math.abs(corr) > 0.5 ? 'Moderate' : 'Weak';
                const direction = corr > 0 ? 'positive' : 'negative';
                const relation = corr > 0 ? 'increases with' : 'decreases with';
                
                insights.push({
                    feature1: labels[i],
                    feature2: labels[j],
                    corr: corr,
                    strength: strength,
                    direction: direction,
                    relation: relation,
                    text: `${strength} ${direction} correlation (r = ${corr.toFixed(3)}) between ${labels[i]} and ${labels[j]}: ${labels[i]} ${relation} ${labels[j]}`
                });
            }
        }
    }
    
    // Sort by absolute correlation strength
    insights.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));
    
    // Add top insights to list
    insights.slice(0, 5).forEach(insight => {
        const li = document.createElement('li');
        li.className = 'mb-1';
        li.innerHTML = `<strong>${insight.feature1} & ${insight.feature2}:</strong> ${insight.text}`;
        insightsList.appendChild(li);
    });
}

// 6. Analyze Survival Factors
function analyzeSurvivalFactors() {
    // Survival by Gender
    analyzeSurvivalByGender();
    
    // Survival by Passenger Class
    analyzeSurvivalByPclass();
    
    // Survival by Age Group
    analyzeSurvivalByAge();
    
    // Calculate and display survival metrics
    calculateSurvivalMetrics();
}

// Analyze survival by gender
function analyzeSurvivalByGender() {
    const genderGroups = {};
    titanicData.forEach(row => {
        if (row.Sex !== null && row.Sex !== undefined && row.Sex !== '') {
            if (!genderGroups[row.Sex]) {
                genderGroups[row.Sex] = { total: 0, survived: 0 };
            }
            genderGroups[row.Sex].total++;
            if (row.Survived === 1) {
                genderGroups[row.Sex].survived++;
            }
        }
    });
    
    const labels = Object.keys(genderGroups);
    const survivalRates = labels.map(gender => 
        (genderGroups[gender].survived / genderGroups[gender].total * 100).toFixed(1)
    );
    
    // Create grouped bar chart
    const ctx = document.getElementById('survivalSexChart').getContext('2d');
    
    if (charts.survivalSexChart) {
        charts.survivalSexChart.destroy();
    }
    
    charts.survivalSexChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(g => g === 'male' ? 'Male' : 'Female'),
            datasets: [
                {
                    label: 'Survived',
                    data: labels.map(gender => genderGroups[gender].survived),
                    backgroundColor: 'rgba(46, 204, 113, 0.7)',
                    borderColor: 'rgb(46, 204, 113)',
                    borderWidth: 1
                },
                {
                    label: 'Perished',
                    data: labels.map(gender => genderGroups[gender].total - genderGroups[gender].survived),
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: 'rgb(231, 76, 60)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Survival by Gender (Female: ${survivalRates[labels.indexOf('female')]}%, Male: ${survivalRates[labels.indexOf('male')]}%)`
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Gender'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Passenger Count'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Analyze survival by passenger class
function analyzeSurvivalByPclass() {
    const classGroups = {};
    titanicData.forEach(row => {
        if (row.Pclass !== null && row.Pclass !== undefined && row.Pclass !== '') {
            if (!classGroups[row.Pclass]) {
                classGroups[row.Pclass] = { total: 0, survived: 0 };
            }
            classGroups[row.Pclass].total++;
            if (row.Survived === 1) {
                classGroups[row.Pclass].survived++;
            }
        }
    });
    
    const labels = Object.keys(classGroups).sort();
    const survivalRates = labels.map(pclass => 
        (classGroups[pclass].survived / classGroups[pclass].total * 100).toFixed(1)
    );
    
    // Create grouped bar chart
    const ctx = document.getElementById('survivalClassChart').getContext('2d');
    
    if (charts.survivalClassChart) {
        charts.survivalClassChart.destroy();
    }
    
    charts.survivalClassChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(c => `Class ${c}`),
            datasets: [
                {
                    label: 'Survived',
                    data: labels.map(pclass => classGroups[pclass].survived),
                    backgroundColor: 'rgba(46, 204, 113, 0.7)',
                    borderColor: 'rgb(46, 204, 113)',
                    borderWidth: 1
                },
                {
                    label: 'Perished',
                    data: labels.map(pclass => classGroups[pclass].total - classGroups[pclass].survived),
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: 'rgb(231, 76, 60)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Survival by Class (1st: ${survivalRates[0]}%, 2nd: ${survivalRates[1]}%, 3rd: ${survivalRates[2]}%)`
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Passenger Class'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Passenger Count'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Analyze survival by age group
function analyzeSurvivalByAge() {
    // Create age groups
    const ageGroups = {
        'Child (0-12)': { min: 0, max: 12, total: 0, survived: 0 },
        'Teen (13-19)': { min: 13, max: 19, total: 0, survived: 0 },
        'Adult (20-39)': { min: 20, max: 39, total: 0, survived: 0 },
        'Middle (40-59)': { min: 40, max: 59, total: 0, survived: 0 },
        'Senior (60+)': { min: 60, max: 120, total: 0, survived: 0 }
    };
    
    // Count passengers in each age group
    titanicData.forEach(row => {
        if (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) {
            for (const [group, range] of Object.entries(ageGroups)) {
                if (row.Age >= range.min && row.Age <= range.max) {
                    ageGroups[group].total++;
                    if (row.Survived === 1) {
                        ageGroups[group].survived++;
                    }
                    break;
                }
            }
        }
    });
    
    const labels = Object.keys(ageGroups);
    const survivalRates = labels.map(group => 
        ageGroups[group].total > 0 ? 
        (ageGroups[group].survived / ageGroups[group].total * 100).toFixed(1) : 
        0
    );
    
    // Create grouped bar chart
    const ctx = document.getElementById('survivalAgeChart').getContext('2d');
    
    if (charts.survivalAgeChart) {
        charts.survivalAgeChart.destroy();
    }
    
    charts.survivalAgeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Survived',
                    data: labels.map(group => ageGroups[group].survived),
                    backgroundColor: 'rgba(46, 204, 113, 0.7)',
                    borderColor: 'rgb(46, 204, 113)',
                    borderWidth: 1
                },
                {
                    label: 'Perished',
                    data: labels.map(group => ageGroups[group].total - ageGroups[group].survived),
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: 'rgb(231, 76, 60)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Survival by Age Group'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Age Group'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Passenger Count'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// Calculate and display survival metrics
function calculateSurvivalMetrics() {
    // Calculate survival rates by different factors
    const metricsContainer = document.getElementById('survivalMetrics');
    metricsContainer.innerHTML = '';
    
    // Gender survival rates
    const malePassengers = titanicData.filter(row => row.Sex === 'male');
    const femalePassengers = titanicData.filter(row => row.Sex === 'female');
    
    const maleSurvived = malePassengers.filter(row => row.Survived === 1).length;
    const femaleSurvived = femalePassengers.filter(row => row.Survived === 1).length;
    
    const maleSurvivalRate = malePassengers.length > 0 ? 
        (maleSurvived / malePassengers.length * 100).toFixed(1) : 0;
    const femaleSurvivalRate = femalePassengers.length > 0 ? 
        (femaleSurvived / femalePassengers.length * 100).toFixed(1) : 0;
    const genderDifferential = (femaleSurvivalRate - maleSurvivalRate).toFixed(1);
    
    // Class survival rates
    const classRates = {};
    [1, 2, 3].forEach(pclass => {
        const classPassengers = titanicData.filter(row => row.Pclass === pclass);
        const classSurvived = classPassengers.filter(row => row.Survived === 1).length;
        classRates[pclass] = classPassengers.length > 0 ? 
            (classSurvived / classPassengers.length * 100).toFixed(1) : 0;
    });
    const classDifferential = (classRates[1] - classRates[3]).toFixed(1);
    
    // Create metric cards
    const metrics = [
        {
            title: 'Gender Gap',
            value: `${genderDifferential}%`,
            description: `Female survival (${femaleSurvivalRate}%) vs Male (${maleSurvivalRate}%)`,
            color: 'primary'
        },
        {
            title: 'Class Gap',
            value: `${classDifferential}%`,
            description: `1st Class (${classRates[1]}%) vs 3rd Class (${classRates[3]}%)`,
            color: 'success'
        },
        {
            title: 'Overall Survival',
            value: `${((titanicData.filter(row => row.Survived === 1).length / titanicData.length) * 100).toFixed(1)}%`,
            description: `${titanicData.filter(row => row.Survived === 1).length} of ${titanicData.length} passengers`,
            color: 'info'
        }
    ];
    
    metrics.forEach(metric => {
        const col = document.createElement('div');
        col.className = 'col-md-4';
        col.innerHTML = `
            <div class="card text-center">
                <div class="card-body">
                    <h6 class="card-subtitle mb-2 text-muted">${metric.title}</h6>
                    <h3 class="card-title text-${metric.color}">${metric.value}</h3>
                    <p class="card-text small">${metric.description}</p>
                </div>
            </div>
        `;
        metricsContainer.appendChild(col);
    });
}

// 7. Generate Insights and Conclusion
function generateInsightsAndConclusion() {
    // Determine the most important survival factor
    const topFactor = determineTopSurvivalFactor();
    document.getElementById('topFactor').textContent = topFactor.factor;
    document.getElementById('factorExplanation').textContent = topFactor.explanation;
    
    // Generate key insights
    const insightsList = document.getElementById('keyInsights');
    insightsList.innerHTML = '';
    
    const insights = [
        `Survival rate was ${((titanicData.filter(row => row.Survived === 1).length / titanicData.length) * 100).toFixed(1)}%, meaning most passengers did not survive.`,
        `Female passengers had a ${((titanicData.filter(row => row.Sex === 'female' && row.Survived === 1).length / 
            titanicData.filter(row => row.Sex === 'female').length) * 100).toFixed(1)}% survival rate vs ${((titanicData.filter(row => row.Sex === 'male' && row.Survived === 1).length / 
            titanicData.filter(row => row.Sex === 'male').length) * 100).toFixed(1)}% for males.`,
        `First class passengers had ${((titanicData.filter(row => row.Pclass === 1 && row.Survived === 1).length / 
            titanicData.filter(row => row.Pclass === 1).length) * 100).toFixed(1)}% survival vs ${((titanicData.filter(row => row.Pclass === 3 && row.Survived === 1).length / 
            titanicData.filter(row => row.Pclass === 3).length) * 100).toFixed(1)}% for third class.`,
        `The dataset has ${columnStats['Age']?.missingPct || 0}% missing Age values and ${columnStats['Cabin']?.missingPct || 0}% missing Cabin values.`,
        `Fare distribution shows significant outliers, with the highest fare being ${Math.max(...titanicData.filter(row => row.Fare).map(row => row.Fare)).toFixed(2)}.`
    ];
    
    insights.forEach(insight => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = insight;
        insightsList.appendChild(li);
    });
    
    // Update conclusion text
    const conclusionText = document.getElementById('conclusionText');
    conclusionText.textContent = `Based on comprehensive exploratory data analysis, ${topFactor.factor} emerges as the most significant factor in determining survival on the Titanic. The analysis reveals clear patterns where ${topFactor.explanation.toLowerCase()} This dashboard provides a complete client-side EDA pipeline that can be deployed to GitHub Pages for easy sharing and collaboration.`;
}

// Determine the most important survival factor
function determineTopSurvivalFactor() {
    // Calculate survival differentials
    const malePassengers = titanicData.filter(row => row.Sex === 'male');
    const femalePassengers = titanicData.filter(row => row.Sex === 'female');
    
    const maleSurvivalRate = malePassengers.length > 0 ? 
        (malePassengers.filter(row => row.Survived === 1).length / malePassengers.length) : 0;
    const femaleSurvivalRate = femalePassengers.length > 0 ? 
        (femalePassengers.filter(row => row.Survived === 1).length / femalePassengers.length) : 0;
    
    const genderDifferential = Math.abs(femaleSurvivalRate - maleSurvivalRate);
    
    // Calculate class differentials
    const classDifferentials = [];
    [1, 2, 3].forEach(pclass => {
        const classPassengers = titanicData.filter(row => row.Pclass === pclass);
        const classSurvivalRate = classPassengers.length > 0 ? 
            (classPassengers.filter(row => row.Survived === 1).length / classPassengers.length) : 0;
        classDifferentials.push(classSurvivalRate);
    });
    
    const classDifferential = Math.max(...classDifferentials) - Math.min(...classDifferentials);
    
    // Determine which factor has the largest differential
    if (genderDifferential > classDifferential) {
        return {
            factor: "Gender (Sex)",
            explanation: "Female passengers had a significantly higher survival rate than male passengers, with a difference of " + 
                (genderDifferential * 100).toFixed(1) + " percentage points."
        };
    } else {
        return {
            factor: "Passenger Class (Pclass)",
            explanation: "First class passengers had much higher survival rates than third class passengers, with a difference of " + 
                (classDifferential * 100).toFixed(1) + " percentage points."
        };
    }
}
