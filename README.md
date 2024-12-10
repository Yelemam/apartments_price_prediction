# apartments_price_prediction
# **Apartment Rental Price Prediction: A Machine Learning Approach**

## **Overview**
This project aims to predict apartment rental prices using machine learning techniques, leveraging enriched datasets and advanced preprocessing. Through Linear Regression, Neural Networks, and Random Forest models, we evaluated predictive performance and optimized outcomes. The project serves as a practical example of how machine learning can address real-world problems in the rental market. The project utilized **Apache Spark** for initial data handling and exploration, efficiently loading and processing the raw dataset before transitioning to Pandas for in-depth analysis and machine learning workflows.

---

## **Dataset**
- **Source**: USA classifieds on the UCI Machine Learning Repository.
- **Description**: Contains 100K records with features like location, price, and amenities. The dataset underwent preprocessing to handle missing values, outliers, and feature transformations for optimal modeling.

**Link to Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)

---

## **Data Preprocessing**
The dataset was first loaded into a Spark DataFrame, enabling efficient handling of the raw CSV file with over 100K records. Spark was used for:
- Parsing the dataset with inferred schema and headers.
- Transitioning to a Pandas DataFrame for further exploration and preprocessing.

### **Steps Taken**:
1. **Handling Missing Data**:
   - Numerical columns: Imputed with mean values.
   - Categorical columns: Filled with "Unknown".
2. **Outlier Treatment**:
   - Capped using the interquartile range (IQR) method.
3. **Feature Engineering**:
   - Expanded `amenities` into binary features.
   - Derived `state` and `city` using geocoding.
   - Transformed date into `year` and `month`.
4. **Feature Selection**:
   - Retained features strongly correlated with `price`.
5. **Data Splitting**:
   - Split into training and testing sets (80/20) to evaluate model performance.

---

## **Models and Results**
### **1. Neural Networks**
- **Enhanced Neural Network**:
  - R-squared: **0.6444**
- **Optimized Neural Network**:
  - R-squared: **0.6117**
- **Insights**:
  - Despite regularization and optimization, performance fell short of the rubric’s threshold.

### **2. Random Forest**
- **Initial Model**:
  - R-squared: **0.8049**
- **Fine-Tuned Model**:
  - R-squared: **0.8053**
  - Hyperparameters tuned:
    - `n_estimators`: 500
    - `max_depth`: 60
- **Outcome**:
  - Random Forest emerged as the best-performing model, surpassing the R-squared benchmark of 0.80.

### **3. Linear Regression**
- R-squared: **0.7256**
- Served as a baseline for comparison, providing valuable insights.

### **4. Gradient Boosting Models**
- **LightGBM**:
  - R-squared: **0.7439**
- **CatBoost**:
  - R-squared: **0.7118**

---

## **Key Insights**
- **Data Enrichment**: Enhanced features like `amenities` and geolocation data contributed significantly to model performance.
- **Hyperparameter Tuning**: Improved Random Forest performance slightly (R-squared 80.53%) through manual fine-tuning.
- **Feature Engineering**: Transformations like one-hot encoding and date splitting boosted model utility.

---

## **Visualizations**
1. **Neural Network Performance**:
   - R-squared comparison charts for Enhanced and Optimized Neural Networks.
2. **Random Forest**:
   - R-squared comparisons for default and fine-tuned models.
3. **Model Comparison**:
   - Visual comparison of Neural Networks, Random Forest, and Gradient Boosting models.

---

## **Installation**
To run the code in Google Colab:
1. Create or open a notebook on the platform.
2. Install any required libraries using `pip install` or `apt-get install`.
3. Upload the necessary CSV files:
   - For **Notebook 1** (`apartments_prediction`), upload the main CSV file: `apartments_for_rent_classified_100K.csv`.
   - For **Notebook 2** (`apartments_data_enrichment_optimization`), upload the cleaned dataset: `apartments_data_cleaned.csv`.
4. Verify installations by importing libraries, then execute your code.
5. Save your work to Google Drive or download it locally, as Colab sessions reset when closed.
---

## Technologies
- **Apache Spark**: For initial data handling and exploration.
- **Pandas**: For in-depth data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For machine learning models and evaluation.
- **Matplotlib & Seaborn**: For visualizations.

## Supplementary Analysis

As part of our exploration, we evaluated additional models to understand their predictive performance. While these models are not incorporated into the main presentation, their results provide valuable insights into alternative machine learning approaches.

### **1. LightGBM**
- **Overview**:
  - LightGBM is an efficient gradient boosting algorithm designed for speed and accuracy, especially with large datasets.
  - It uses histogram-based learning, enabling fast training with sparse features.

- **Results**:
  - **Mean Squared Error (MSE)**: 170,490.74
  - **R-squared**: 0.7439 (74.39%)

---

### **2. CatBoost**
- **Overview**:
  - CatBoost excels with categorical features and requires minimal preprocessing, making it highly efficient for real-world datasets.
  - It automatically handles categorical data without extensive feature engineering.

- **Results**:
  - **Mean Squared Error (MSE)**: 191,902.68
  - **R-squared**: 0.7118 (71.18%)

---

### **Insights from Exploration**
- While **LightGBM** achieved better results (R-squared: 74.39%) compared to **CatBoost** (R-squared: 71.18%), both underperformed relative to the **Random Forest** model, which remains the best-performing approach.
- These exploratory results highlight the robustness of Random Forest and its ability to handle nonlinear relationships effectively.
- The additional exploration underscores the importance of experimenting with multiple models to validate performance.

---

### Note:
The results from LightGBM and CatBoost are supplementary and were not included in the presentation due to time constraints.


---

## **Roadmap**
Although the project is complete, there is always room for improvement. In the future, we may:
- Optimize existing features.
- Refine model performance.
- Implement additional models to enhance the project’s capabilities.

We are open to evolving the project and welcome any contributions or suggestions for future enhancements.

---

## **Contributing**
We’re always excited to welcome new contributors! If you'd like to help improve this project:
1. Fork the repository.
2. Submit a pull request with your changes.
3. Ensure your contributions follow our coding standards and include detailed descriptions of your updates.

Thank you for helping make this project better!

---

## **Acknowledgments**
We want to express our sincere gratitude to all the contributors who have helped bring this project to life. Your collaboration, innovation, and hard work have been instrumental in its success. Whether through code, ideas, bug fixes, or feedback, each of you has made a significant impact, and we couldn’t have done it without you.

Thank you for being part of this journey!

---

## **Contributors**
Chinna Maijala, Kimberly Her, Yara El-Emam, Zane Huttinga

---

## **Conclusion**
This project demonstrates the importance of data quality, enrichment, and tuning in achieving predictive accuracy. Random Forest remains the most reliable model, but future work could explore stacking or other ensemble techniques to leverage the strengths of multiple algorithms.
