// app.js
/**
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
const dataPreview = document.getElementById('dataPreview');
const dataTypesTable = document.getElementById('dataTypesTable');
const outlierTable = document.getElementById('outlierTable');
const survivalStatsTable = document.getElementById('survivalStatsTable');

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
    if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file.');
        return;
    }
    
    showLoading(true);
    
    // Parse CSV using PapaParse
    Papa.parse(file, {
        header: true,
        dynamicTyping: true, // Automatically convert numbers
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
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
            if (row[col] === null || row[col] === undefined || row[col] === '') {
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
    const previewBody = dataPreview.querySelector('tbody');
    previewBody.innerHTML = '';
    
    // Show first 5 rows
    const previewRows = titanicData.slice(0, 5);
    
    previewRows.forEach((row, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${row.PassengerId || ''}</td>
            <td>${row.Survived !== undefined ? row.Survived : ''}</td>
            <td>${row.Pclass || ''}</td>
            <td class="text-truncate" style="max-width: 150px;">${row.Name || ''}</td>
            <td>${row.Sex || ''}</td>
            <td>${row.Age !== undefined ? row.Age.toFixed(1) : ''}</td>
            <td>${row.SibSp || ''}</td>
            <td>${row.Parch || ''}</td>
            <td>${row.Ticket || ''}</td>
            <td>${row.Fare !== undefined ? row.Fare.toFixed(2) : ''}</td>
            <td>${row.Cabin || ''}</td>
            <td>${row.Embarked || ''}</td>
        `;
        previewBody.appendChild(tr);
    });
}

/**
 * Update data types and missing values table
 */
function updateDataTypesTable() {
    const tableBody = dataTypesTable.querySelector('tbody');
    tableBody.innerHTML = '';
    
    if (!titanicData || titanicData.length === 0) return;
    
    const columns = Object.keys(titanicData[0]);
    
    columns.forEach(col => {
        // Determine data type
        const sampleValue = titanicData.find(row => row[col] !== null && row[col] !== undefined && row[col] !== '');
        let dataType = 'Unknown';
        
        if (sampleValue !== undefined) {
            dataType = typeof sampleValue[col];
            if (dataType === 'number') {
                // Check if it's integer or float
                dataType = Number.isInteger(sampleValue[col]) ? 'Integer' : 'Float';
            } else {
                dataType = dataType.charAt(0).toUpperCase() + dataType.slice(1);
            }
        }
        
        // Count missing values
        const missingCount = titanicData.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === ''
        ).length;
        
        const missingPercent = ((missingCount / titanicData.length) * 100).toFixed(1);
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${col}</strong></td>
            <td><span class="badge bg-info">${dataType}</span></td>
            <td>${missingCount}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${missingPercent > 20 ? 'bg-danger' : 'bg-warning'}" 
                         role="progressbar" 
                         style="width: ${missingPercent}%">
                        ${missingPercent}%
                    </div>
                </div>
            </td>
        `;
        tableBody.appendChild(tr);
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
            row[col] === null || row[col] === undefined || row[col] === ''
        ).length;
        return ((missingCount / titanicData.length) * 100).toFixed(1);
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
                    'rgba(46, 204, 113, 0.8)'
                ),
                borderColor: missingPercentages.map(p => 
                    p > 50 ? 'rgba(231, 76, 60, 1)' : 
                    p > 20 ? 'rgba(241, 196, 15, 1)' : 
                    'rgba(46, 204, 113, 1)'
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
                            return `${context.parsed.x}% missing`;
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
    
    // Run analysis in sequence
    setTimeout(() => {
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
        
        showLoading(false);
        
        // Scroll to insights section
        document.getElementById('insights').scrollIntoView({ behavior: 'smooth' });
    }, 500);
}

/**
 * Create distribution charts for numerical and categorical variables
 */
function createDistributionCharts() {
    // 1. Age Distribution
    createHistogramChart('ageDistributionChart', 'Age Distribution', 
        titanicData.map(p => p.Age).filter(age => !isNaN(age)), 
        'Age (years)', 'Passenger Count', 15);
    
    // 2. Fare Distribution
    createHistogramChart('fareDistributionChart', 'Fare Distribution', 
        titanicData.map(p => p.Fare).filter(fare => !isNaN(fare) && fare < 200), // Filter extreme outliers for better visualization
        'Fare (£)', 'Passenger Count', 20);
    
    // 3. Family Size Distribution (SibSp + Parch)
    const familySizes = titanicData.map(p => (p.SibSp || 0) + (p.Parch || 0));
    createHistogramChart('familyDistributionChart', 'Family Size Distribution', 
        familySizes, 'Family Size (SibSp + Parch)', 'Passenger Count', 10);
    
    // 4. Gender Distribution
    createPieChart('genderChart', 'Passenger Gender', 
        ['Male', 'Female'], 
        [
            titanicData.filter(p => p.Sex === 'male').length,
            titanicData.filter(p => p.Sex === 'female').length
        ],
        ['#3498db', '#e74c3c']);
    
    // 5. Passenger Class Distribution
    createPieChart('classChart', 'Passenger Class', 
        ['1st Class', '2nd Class', '3rd Class'], 
        [
            titanicData.filter(p => p.Pclass === 1).length,
            titanicData.filter(p => p.Pclass === 2).length,
            titanicData.filter(p => p.Pclass === 3).length
        ],
        ['#2ecc71', '#f1c40f', '#e74c3c']);
    
    // 6. Embarkation Port Distribution
    const embarkedCounts = {};
   
