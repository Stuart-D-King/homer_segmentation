## Microgrid Customer Segmentation  
### Problem Statement
[HOMER Energy](http://www.homerenergy.com/) is a world leader in accurately modeling microgrid optimization. The HOMER (Hybrid Optimization of Multiple Energy Resources) software allows users to understand how to build cost effective and reliable microgrids that combine traditionally generated and renewable power, storage, and load management. HOMER Energy also offers software training, microgrid consulting expertise, and market access support to its vendor partners.

As the microgrid market continues to expand, HOMER and its vendor partners needs to improve their shared business intelligence related to consumer behavior and demand. Therefore, the objective of this study is to extract structure and meaning from HOMER Energy's collection of user software simulations. By being able to segment its customer base, HOMER can enhance the market access branch of its business model by providing its vendor partners with more reliable information related to the micrgrid consumer market. Understanding which software users more likely to initialize micogrids is information all vendors want to know to better target those customers ready to get started.

### Staged Approach
Phase 1: Problem Framing - Complete
- Meet and consult with HOMER Energy counterpart
- Identify need
- Succinctly frame problem
- Outline approach and data pipeline  
- **Work product:** Clear project objective

Phase 2: Data Munging - Current
- Determine strategy for missing values
- Select initial features and identify areas of data sparsity
- Subset data into user matrix form  
- **Work product:** Dense data frame suitable for EDA and modeling

Phase 3: EDA and Modeling
- Apply unsupervised clustering algorithms to data frame
- Visualize data subsets to identify trends
- Identify additional data sources to incorporate into model  
- **Work product:** Initial models, visualizations, additional data

Phase 4: Final Model and Feature Importance
- Run k-prototypes and/or k-modes clustering clustering algorithms
- Produce dendrograms, scree plots, and 2-d PCA for cluster presentation
- Use dimensionality reduction methods to extract feature importance  
- **Work product:** Working model with supporting visualizations to infer business intelligence

Phase 5: Web Application and Presentation
- Build Flask application to deploy on Amazon Web Services
- Present findings and application at Galvanize and HOMER Energy
- Outline next steps

### References
Academic article on the extension of the k-means algorithm to categorical data:  
http://arbor.ee.ntu.edu.tw/~chyun/dmpaper/huanet98.pdf  

Python implementation of k-modes and k-prototypes clustering algorithms for categorical data:  
https://github.com/nicodv/kmodes#id1  

Example of customer segmentation using unsupervised learning techniques:   http://www.ritchieng.com/machine-learning-project-customer-segments/  

Another example of customer segmentation in Python:  
http://blog.yhat.com/posts/customer-segmentation-using-python.html
