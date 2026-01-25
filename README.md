# nndl
**Prompt**

Role: You are a senior front-end engineer and UI/UX designer specializing in building polished, interactive data dashboards for browser-based analysis.

Overall Task: Build a beautifully-styled, single-page EDA dashboard app that runs entirely in the browser and is deployable to GitHub Pages. The app will perform a thorough Exploratory Data Analysis on the Kaggle Titanic dataset (https://www.kaggle.com/datasets/yasserh/titanic-dataset?resource=download), loaded from a single file ('Titanic-Dataset.csv'). The analysis must investigate: data distributions, missing values, outliers, correlations, underlying patterns, data types, and comprehensive visualizations. The dashboard must culminate in identifying key insights and determining the single most important factor associated with passenger survival (all based on analysis).

Strictly output in two separate code blocks:

1.  Label the first as 'index.html' (for the complete HTML structure, integrated CSS for beautiful styling, and framework/CDN links).
2.  Label the second as 'app.js' (for all JavaScript logic, data processing, and Chart.js interactions).

Use the following CDN libraries: PapaParse (for CSV parsing), Chart.js (for all visualizations), and Bootstrap 5 (for responsive layout and styling components). All processing must be client-side. Link 'app.js' from ‘index.html' !!! Make sure all the buttons in the app work properly and backed runs correctly !!! 

Workflow:

In 'index.html': Create a visually cohesive dashboard with a clear navigation header and the following card-based sections: 1) Data Load & Overview, 2) Data Types & Missing Values, 3) Distribution Analysis (Numerical & Categorical), 4) Outlier Detection, 5) Correlation & Pattern Analysis, 6) Survival Factor Investigation, 7) Key Insights & Conclusion. Include a file input for 'Titanic-Dataset.csv'. The styling should be elegant and modern using Bootstrap 5 utilities and custom CSS. Include a deployment note at the end: "Push to a public GitHub repo, enable Pages (main branch /root), and visit your '<username>.github.io/<repo-name>' URL."

In 'app.js':  
- Load Data: Use PapaParse to load the CSV. Implement robust error handling (missing file, parse errors) with user-friendly alerts.
- Overview & Data Types: Display a preview table, dataset shape (rows/columns), and a clean table listing each column with its data type and missing value count/percentage.
- Missing Data & Cleaning: Render a horizontal bar chart (Chart.js) showing the % of missing values per column. Provide a summary statement.
- Distributions: Create: a) Histograms for 'Age', 'Fare', 'SibSp', 'Parch'; b) Bar charts for 'Sex', 'Pclass', 'Embarked', 'Survived'.
- Outlier Detection: For key numerical columns ('Age', 'Fare') identify outliers. Present counts and visualize 'Fare' outliers using a box plot chart.
- Correlations: Calculate a correlation matrix for numerical fields ('Age', 'Fare', 'Pclass', 'SibSp', 'Parch'). Render a visually clear correlation heatmap using a Chart.js bar chart with a color gradient.
- Survival Factor Analysis: Systematically analyze and visualize survival rates by 'Sex', 'Pclass', and 'Age' group. Calculate and compare metrics (e.g., survival percentage differentials) to programmatically identify and state the most important survival factor in a dedicated conclusion card.
- Interactivity: Make the dashboard intuitive and guide the user. All the buttons should be clickable. Users should be able to upload a CSV file into the app and press “Run EDA” button to start the EDA pipeline. 

The dashboard color palette should be in muted blue tones. I am going to deploy this app on GitHub Pages: “nndl” repository, “week1” directory. Add clear code comments in English.

**Extra Prompt**
After I deployed this on github pages Fare Outliers Box Plot and the data in sections 5. Correlation & Pattern Analysis, 6. Survival Factor Investigation, 7. Key Insights & Conclusion is not displayed. It seems there is something wrong with js file.
