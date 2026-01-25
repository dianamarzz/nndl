// app.js - Titanic EDA Dashboard
// Comprehensive client-side EDA for the Titanic dataset

// Global variables to store data and analysis results
let rawData = [];
let cleanedData = [];
let analysisResults = {
    overview: {},
    dataQuality: {},
    univariate: {},
    bivariate: {},
    outliers: {},
    insights: []
};

// Chart instances for easy updates and destruction
const chartInstances = {};

// Main initialization function
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

/**
 * Initialize all event listeners for the dashboard
 */
function initializeEventListeners() {
    // Load and analyze data button
    document.getElementById('loadDataBtn').addEventListener('click', loadAndAnalyzeData);
    
    // Generate report button
    document.getElementById('generateReportBtn').addEventListener('click', generateFullReport);
    
    // Export buttons
    document.getElementById('exportJsonBtn').addEventListener('click', exportJson);
    document.getElementById('exportCleanedCsvBtn').addEventListener('click', exportCleanedCsv);
}

/**
 * Main function to load CSV data and perform EDA
 */
function loadAndAnalyzeData() {
    const fileInput = document.getElementById('csvFile');
    
    if (!fileInput.files.length) {
        showStatus('Please select a CSV file to upload.', 'danger');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Check if filename matches expected pattern
    if (!file.name.toLowerCase().includes('titanic')) {
        if (!confirm('The file name doesn\'t appear to be the Titanic dataset. Continue anyway?')) {
            return;
        }
    }
    
    showLoading(true);
    showStatus('Loading and parsing CSV file...', 'info');
    
    // Parse CSV with PapaParse
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            if (results.errors.length > 0) {
                showStatus(`Error parsing CSV: ${results.errors[0].message}`, 'danger');
                showLoading(false);
                return;
            }
            
            rawData = results.data;
            showStatus(`Successfully loaded ${rawData.length} rows with ${Object.keys(rawData[0]).length} columns`, 'success');
            
            // Perform comprehensive EDA
            performEDA();
            
            // Enable report generation and export buttons
            document.getElementById('generateReportBtn').disabled = false;
            document.getElementById('exportJsonBtn').disabled = false;
            document.getElementById('exportCleanedCsvBtn').disabled = false;
            
            showLoading(false);
        },
        error: function(error) {
            showStatus(`Error loading file: ${error.message}`, 'danger');
            showLoading(false);
        }
    });
}

/**
 * Perform comprehensive Exploratory Data Analysis
 */
function performEDA() {
    // Clean the data first
    cleanedData = cleanData(rawData);
    
    // Perform all analyses
    analyzeDataOverview();
    analyzeDataQuality();
    performUnivariateAnalysis();
    performBivariateAnalysis();
    detectOutliers();
    generateInsights();
    
    // Update the UI with results
    updateDashboard();
}

/**
 * Clean the dataset: handle missing values and data inconsistencies
 */
function cleanData(data) {
    // Create a deep copy of the data
    const cleaned = JSON.parse(JSON.stringify(data));
    
    // Define strategies for missing values
    const numericColumns = ['Age', 'Fare', 'SibSp', 'Parch'];
    const categoricalColumns = ['Embarked', 'Cabin'];
    
    // Calculate median for numeric columns
    const medians = {};
    numericColumns.forEach(col => {
        const values = cleaned.map(row => row[col]).filter(val => val != null && !isNaN(val));
        if (values.length > 0) {
            values.sort((a, b) => a - b);
            const mid = Math.floor(values.length / 2);
            medians[col] = values.length % 2 === 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
        } else {
            medians[col] = 0;
        }
    });
    
    // Calculate mode for categorical columns
    const modes = {};
    categoricalColumns.forEach(col => {
        const valueCounts = {};
        cleaned.forEach(row => {
            if (row[col] != null && row[col] !== '') {
                valueCounts[row[col]] = (valueCounts[row[col]] || 0) + 1;
            }
        });
        
        let maxCount = 0;
        let mode = null;
        Object.keys(valueCounts).forEach(key => {
            if (valueCounts[key] > maxCount) {
                maxCount = valueCounts[key];
                mode = key;
            }
        });
        modes[col] = mode || 'Unknown';
    });
    
    // Apply cleaning strategies
    cleaned.forEach(row => {
        // Fill numeric missing values with median
        numericColumns.forEach(col => {
            if (row[col] == null || isNaN(row[col])) {
                row[col] = medians[col];
            }
        });
        
        // Fill categorical missing values with mode
        categoricalColumns.forEach(col => {
            if (row[col] == null || row[col] === '') {
                row[col] = modes[col];
            }
        });
        
        // Ensure Survived is boolean (0 or 1)
        if (row.Survived != null) {
            row.Survived = row.Survived ? 1 : 0;
        }
    });
    
    return cleaned;
}

/**
 * Analyze dataset overview: shape, preview, basic stats
 */
function analyzeDataOverview() {
    if (cleanedData.length === 0) return;
    
    const columns = Object.keys(cleanedData[0]);
    const totalRows = cleanedData.length;
    const totalColumns = columns.length;
    
    // Calculate survival rate
    const survivedCount = cleanedData.filter(row => row.Survived === 1).length;
    const survivalRate = ((survivedCount / totalRows) * 100).toFixed(1);
    
    // Store overview results
    analysisResults.overview = {
        shape: `${totalRows} rows × ${totalColumns} columns`,
        survivalRate: `${survivalRate}% (${survivedCount}/${totalRows})`,
        features: columns.join(', '),
        columns: columns,
        totalRows: totalRows,
        survivedCount: survivedCount
    };
}

/**
 * Analyze data quality: missing values and data types
 */
function analyzeDataQuality() {
    if (rawData.length === 0) return;
    
    const columns = Object.keys(rawData[0]);
    const missingData = {};
    const dataTypes = {};
    const uniqueValues = {};
    
    columns.forEach(col => {
        // Calculate missing values
        const missingCount = rawData.filter(row => 
            row[col] == null || row[col] === '' || (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        const missingPercent = ((missingCount / rawData.length) * 100).toFixed(1);
        
        missingData[col] = {
            count: missingCount,
            percent: parseFloat(missingPercent)
        };
        
        // Determine data type
        const sampleValues = rawData.map(row => row[col]).filter(val => val != null);
        let type = 'Unknown';
        
        if (sampleValues.length > 0) {
            const firstType = typeof sampleValues[0];
            if (firstType === 'number') {
                type = 'Numeric';
            } else if (firstType === 'string') {
                type = 'String';
            } else if (firstType === 'boolean') {
                type = 'Boolean';
            }
        }
        
        dataTypes[col] = type;
        
        // Count unique values
        const uniqueSet = new Set(sampleValues.map(val => String(val)));
        uniqueValues[col] = uniqueSet.size;
    });
    
    analysisResults.dataQuality = {
        missingData: missingData,
        dataTypes: dataTypes,
        uniqueValues: uniqueValues
    };
}

/**
 * Perform univariate analysis: distributions of individual features
 */
function performUnivariateAnalysis() {
    if (cleanedData.length === 0) return;
    
    // Categorical features distributions
    const categoricalFeatures = ['Pclass', 'Sex', 'Embarked'];
    const categoricalDistributions = {};
    
    categoricalFeatures.forEach(feature => {
        const counts = {};
        cleanedData.forEach(row => {
            const value = row[feature];
            if (value != null) {
                const key = String(value);
                counts[key] = (counts[key] || 0) + 1;
            }
        });
        
        // Convert to arrays for Chart.js
        const labels = Object.keys(counts);
        const data = Object.values(counts);
        
        categoricalDistributions[feature] = { labels, data };
    });
    
    // Numeric features distributions
    const numericFeatures = ['Age', 'Fare', 'SibSp', 'Parch'];
    const numericDistributions = {};
    
    numericFeatures.forEach(feature => {
        const values = cleanedData.map(row => row[feature]).filter(val => val != null);
        
        // Calculate histogram bins
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binCount = 15;
        const binWidth = (max - min) / binCount;
        
        const bins = Array(binCount).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binWidth;
            const binEnd = binStart + binWidth;
            binLabels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}`);
            
            values.forEach(val => {
                if (val >= binStart && (i === binCount - 1 ? val <= binEnd : val < binEnd)) {
                    bins[i]++;
                }
            });
        }
        
        // Calculate summary statistics
        const sorted = [...values].sort((a, b) => a - b);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const median = sorted[Math.floor(sorted.length / 2)];
        
        numericDistributions[feature] = {
            labels: binLabels,
            data: bins,
            stats: {
                min: min,
                max: max,
                mean: mean.toFixed(2),
                median: median.toFixed(2),
                std: calculateStandardDeviation(values).toFixed(2)
            }
        };
    });
    
    analysisResults.univariate = {
        categorical: categoricalDistributions,
        numeric: numericDistributions
    };
}

/**
 * Perform bivariate analysis: relationships between features and survival
 */
function performBivariateAnalysis() {
    if (cleanedData.length === 0) return;
    
    // Survival by categorical features
    const survivalByFeature = {};
    const featuresToAnalyze = ['Pclass', 'Sex', 'Embarked'];
    
    featuresToAnalyze.forEach(feature => {
        const values = [...new Set(cleanedData.map(row => row[feature]))];
        const survivalRates = [];
        
        values.forEach(value => {
            const group = cleanedData.filter(row => row[feature] === value);
            const survived = group.filter(row => row.Survived === 1);
            const survivalRate = group.length > 0 ? (survived.length / group.length) * 100 : 0;
            survivalRates.push(survivalRate.toFixed(1));
        });
        
        survivalByFeature[feature] = {
            labels: values.map(String),
            rates: survivalRates.map(parseFloat),
            counts: values.map(value => cleanedData.filter(row => row[feature] === value).length)
        };
    });
    
    // Survival by age groups
    const ageGroups = ['0-12', '13-25', '26-40', '41-60', '60+'];
    const ageGroupRates = [];
    
    ageGroups.forEach(group => {
        let minAge, maxAge;
        if (group === '0-12') {
            minAge = 0; maxAge = 12;
        } else if (group === '13-25') {
            minAge = 13; maxAge = 25;
        } else if (group === '26-40') {
            minAge = 26; maxAge = 40;
        } else if (group === '41-60') {
            minAge = 41; maxAge = 60;
        } else {
            minAge = 61; maxAge = 200;
        }
        
        const groupData = cleanedData.filter(row => row.Age >= minAge && row.Age <= maxAge);
        const survived = groupData.filter(row => row.Survived === 1);
        const rate = groupData.length > 0 ? (survived.length / groupData.length) * 100 : 0;
        ageGroupRates.push(rate.toFixed(1));
    });
    
    survivalByFeature['AgeGroup'] = {
        labels: ageGroups,
        rates: ageGroupRates.map(parseFloat)
    };
    
    // Correlation matrix
    const numericColumns = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'];
    const correlationMatrix = calculateCorrelationMatrix(numericColumns);
    
    analysisResults.bivariate = {
        survivalByFeature: survivalByFeature,
        correlationMatrix: correlationMatrix,
        numericColumns: numericColumns
    };
}

/**
 * Detect outliers using the IQR method
 */
function detectOutliers() {
    if (cleanedData.length === 0) return;
    
    const numericColumns = ['Age', 'Fare'];
    const outlierResults = {};
    
    numericColumns.forEach(column => {
        const values = cleanedData.map(row => row[column]).filter(val => val != null);
        
        // Calculate quartiles
        const sorted = [...values].sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        
        // Define outlier boundaries
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        
        // Count outliers
        const outliers = values.filter(val => val < lowerBound || val > upperBound);
        const outlierPercent = (outliers.length / values.length) * 100;
        
        outlierResults[column] = {
            total: values.length,
            outliers: outliers.length,
            percent: outlierPercent.toFixed(2),
            lowerBound: lowerBound.toFixed(2),
            upperBound: upperBound.toFixed(2),
            q1: q1.toFixed(2),
            q3: q3.toFixed(2),
            iqr: iqr.toFixed(2)
        };
    });
    
    analysisResults.outliers = outlierResults;
}

/**
 * Generate insights and identify the most important factor for survival
 */
function generateInsights() {
    if (cleanedData.length === 0) return;
    
    const insights = [];
    
    // Insight 1: Overall survival rate
    const totalPassengers = cleanedData.length;
    const survivedCount = cleanedData.filter(row => row.Survived === 1).length;
    const survivalRate = ((survivedCount / totalPassengers) * 100).toFixed(1);
    insights.push(`Overall survival rate was ${survivalRate}% (${survivedCount} of ${totalPassengers} passengers)`);
    
    // Insight 2: Survival by gender
    const femalePassengers = cleanedData.filter(row => row.Sex === 'female');
    const malePassengers = cleanedData.filter(row => row.Sex === 'male');
    const femaleSurvivalRate = femalePassengers.length > 0 ? 
        ((femalePassengers.filter(row => row.Survived === 1).length / femalePassengers.length) * 100).toFixed(1) : 0;
    const maleSurvivalRate = malePassengers.length > 0 ? 
        ((malePassengers.filter(row => row.Survived === 1).length / malePassengers.length) * 100).toFixed(1) : 0;
    insights.push(`Female passengers had a ${femaleSurvivalRate}% survival rate, compared to ${maleSurvivalRate}% for male passengers`);
    
    // Insight 3: Survival by class
    const classSurvival = {};
    [1, 2, 3].forEach(pclass => {
        const classPassengers = cleanedData.filter(row => row.Pclass === pclass);
        const classSurvived = classPassengers.filter(row => row.Survived === 1).length;
        classSurvival[pclass] = (classSurvived / classPassengers.length * 100).toFixed(1);
    });
    insights.push(`First-class passengers had the highest survival rate at ${classSurvival[1]}%, followed by second class (${classSurvival[2]}%) and third class (${classSurvival[3]}%)`);
    
    // Insight 4: Age and survival
    const children = cleanedData.filter(row => row.Age < 18);
    const adults = cleanedData.filter(row => row.Age >= 18);
    const childSurvivalRate = children.length > 0 ? 
        ((children.filter(row => row.Survived === 1).length / children.length) * 100).toFixed(1) : 0;
    const adultSurvivalRate = adults.length > 0 ? 
        ((adults.filter(row => row.Survived === 1).length / adults.length) * 100).toFixed(1) : 0;
    insights.push(`Children (under 18) had a ${childSurvivalRate}% survival rate, while adults had ${adultSurvivalRate}%`);
    
    // Insight 5: Family size and survival
    cleanedData.forEach(row => {
        row.FamilySize = row.SibSp + row.Parch + 1;
    });
    
    const soloPassengers = cleanedData.filter(row => row.FamilySize === 1);
    const familyPassengers = cleanedData.filter(row => row.FamilySize > 1);
    const soloSurvivalRate = soloPassengers.length > 0 ? 
        ((soloPassengers.filter(row => row.Survived === 1).length / soloPassengers.length) * 100).toFixed(1) : 0;
    const familySurvivalRate = familyPassengers.length > 0 ? 
        ((familyPassengers.filter(row => row.Survived === 1).length / familyPassengers.length) * 100
