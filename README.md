# Classification of Performance and Absenteeism in ENEM

## About the Project
This project analyzes the influence of socioeconomic factors on the performance and absenteeism of participants in the Brazilian National High School Exam (ENEM) between 2019 and 2023. It is characterized as a quantitative analytical study that uses Machine Learning techniques to identify patterns of social inequality reflected in education. The research uses open microdata provided by the National Institute for Educational Studies and Research Anísio Teixeira (INEP).

## Authors
- Micael Oliveira de Lima Toscano  
- Sérgio Cauã dos Santos  

**Advisors:** Prof. Dr. Bruno Jefferson de Sousa Pessoa and Prof. Dr. Gilberto Farias de Sousa Filho  
**Institution:** Federal University of Paraíba (UFPB) - Center for Informatics  

## Objectives
- Apply Machine Learning models to classify student performance based on their socioeconomic data  
- Estimate the probability of participant attendance on exam days  
- Build an inference pipeline based on these probabilities to generate educational risk profiles for students  
- Use performance predictions as a score to estimate relative chances of admission to higher education  

## Technologies and Libraries Used
- **Machine Learning Models:** Decision Tree, Random Forest, Support Vector Machine (SVM), Neural Networks  
- **Data Manipulation:** Pandas, NumPy  
- **Modeling and Evaluation:** Scikit-learn, Keras  
- **Visualization:** Matplotlib  

## Methodology
1. **Data Engineering:** Due to the large volume of information, the original INEP data was converted from `.csv` to `.parquet`, ensuring greater efficiency and faster read/compression operations  
2. **Preprocessing:** Applied techniques such as One-Hot Encoding for categorical variables and Ordinal Encoding to preserve order relationships. Sensitive columns (e.g., race and gender) were removed to reduce bias. The target variable `FALTOU` was created to represent attendance  
3. **Training and Validation:** Models were evaluated using cross-validation to ensure generalization. Temporal splits were tested, using data from 2019–2022 for training/validation and 2023 exclusively for testing  

## Results
- **Performance Classification:** Neural Networks achieved the best performance, with **71% accuracy**  
- **Attendance Classification (Absenteeism):** Tree-based models (Decision Tree and Random Forest) performed better and provided high interpretability  
- **Conclusion:** Socioeconomic data proved effective in predicting student behavior (~70% accuracy), enabling the identification of vulnerable profiles even before the exam
