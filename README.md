## Microgrid Customer Segmentation  
### Problem Statement
[HOMER Energy](http://www.homerenergy.com/) is a world leader in accurately modeling microgrid optimization. The HOMER (Hybrid Optimization of Multiple Energy Resources) software allows users to understand how to build cost effective and reliable microgrids that combine traditionally generated and renewable power, storage, and load management. HOMER Energy also offers software training, microgrid consulting expertise, and market access support to its vendor partners.

As the microgrid market continues to expand, HOMER and its vendor partners needs to better understand its software users' behavior and intensions. The objective of this study is to extract structure and meaning from HOMER Energy's collection of user software simulations. By being able to segment its customer base, HOMER can enhance the market access branch of its business model by providing its vendor partners with more reliable information related to the micrgrid consumer market. Understanding which software users more likely to initialize micogrids is information all vendors want to know to better target those customers ready to get started.

### Process
#### Data Preprocessing
The first and very important step in my process was to understand the data and prepare it for modeling. The HOMER software is extremely powerful and capable of modeling hundreds of microgird configurations at once. Because the software has predictive capabilities built into it, my task was to understand which features in the data space would be insightful for determining the realness or seriousness of a project. Initial features were selected based on consultations with my HOMER counterpart, Jordan Esbin, as well as my participation in HOMER's online software training series. After selecting the appropriate user input features, I performed a variety of data munging and cleaning steps using the Pandas data analysis library. For example, I bucketed the User Role and Organization Type variables into categories such as academic, business, technical, and undefined, and categorized whether or not a simulation modeled certain types of energy hardware. This extensive data cleaning step reduced the features space from 144 variables to 15. Final model features include:

| Feature   | Description | Data Type |
| -------   | ----------- | --------- |
| User | Six-digit numeric ID for each user | String |
| UserRole | User identified professional occupation | String |
| OrganizationType | User identified employment field | String |
| Latitude | Latitude coordinate of simulated project | Float |
| Longitude | Longitude coordinate of simulated project | Float |
| MultiGenSearch | If applicable, if the user simulated a variety of generator models | Category |
| MultiWindSearch | If applicable, if the user simulated a variety of wind turbine models | Category |
| MultiBatSearch | If applicable, if the user simulated a variety of battery quantities | Category |
| MultiPvSearch | If applicable, if the user simulated a variety of solar panel configurations | Category |
| MultiConSearch | If applicable, if the user simulated a variety of converter configurations | Category |
| Sample | Whether a sample file was used in the simulation | Boolean |
| DefaultGenerator | If applicable, if the user used the default generator model in the simulation | Category |
| ImportedWind | If the user opted to import wind data in lieu of using the default parameters | Boolean |
| ImportedSolar | If the user opted to import solar data in lieu of using the default parameters | Boolean |
| Country | Country of the simulated project | Category |

### Model Development
Because HOMER does not currently have a way of tracking which of its software users have become microgrid implementers, I explored various unsupervised clustering algorithms to model the underlying structure of the data. KMeans is a common clustering algorithm that groups data according to existing similarities. However, because the similarity metric used in KMeans is the computed euclidean distance between each data point and the centroids of each cluster, the KMeans algorithm is not appropriate for non-numerical data. A more appropriate clustering algorithm is **KModes**. KModes is an extension of KMeans, however, instead of calculating distance, it quantifies the total number of mismatched categories between two objects: the small the number, the more similar the two objects. In addition, KModes uses modes instead of means, in which the mode is a vector of elements that minimizes the dissimilarities between the vector and an individual data point.

Other unsupervised clustering algorithms were also tested on the data, including **hierarchical agglomerative clustering** and **gaussian mixture models**. To test these algorithms, I used one-hot encoding to further transform the feature space into binary/dummy variables. Both algorithms produced clustered results, however, the interpretability of the clusters was compromised due to the inherent non-numerical nature of the data. For this reason, I settled on KModes as the algorithm to formulate my findings. You can read more about KModes clustering in Zhexue Huang's research paper on the topic.

Once the cluster algorithm was selected, I needed to determine the appropriate amount of clusters to segment the data. In KMeans clustering, it is common to perform a dimensionality reduction technique such as principle component analysis (PCA) to plot the resulting clusters in a two dimensional space to visually confirm clustering is taking place. However, because applying PCA to categorical data is generally regarded as unwise, I instead opted to calculate a silhouette score based on a hamming distance metric for a range of cluster quantities to settle on the most appropriate number of customer segments. When calculating the silhouette coefficient, the best possible value is 1, indicating all data points are perfected grouped and there are no overlapping clusters. A coefficient of 0 implies overlapping clusters, and negative values suggest that a sample has been assigned to the wrong cluster. Based on the below graphics, I decided to cluster my data into four clusters. I selected k=4 because the silhouette score was only slightly lower than k=3, and based on my conversations with HOMER four customer segments fit aligned closely with expected user types. Furthermore, while the silhouette score for k=7 was also similar, I ultimately decided an analysis of seven customer segments would be more appropriate for future analysis.

![silhouette_vs_k](img/silhouette/sil_v_clust_KM_users.png)

After selecting the appropriate number of clusters, I fitted the KModes clustering algorithm to my data and assigned cluster labels to each user. The cluster labels were then mapped back to the full dataset of all simulations using the user's ID as the pairing key. I then performed my initial exploratory data analysis to begin extracting meaningful business intelligence from the clusters.

### Staged Approach
Phase 1: Problem Framing - Complete
- Meet and consult with HOMER Energy counterpart
- Identify need
- Succinctly frame problem
- Outline approach and data pipeline  
- **Work product:** Clear project objective

Phase 2: Data Munging - Complete (may revisit)
- Determine strategy for missing values
- Select initial features and identify areas of data sparsity
- Subset data into user matrix form  
- **Work product:** Dense data frame suitable for EDA and modeling

Phase 3: EDA and Modeling - Current
- Apply unsupervised clustering algorithms to data frame
- Visualize data subsets to identify trends
- Identify additional data sources to incorporate into model  
- **Work product:** Initial models, visualizations, additional data

Phase 4: Final Model and Feature Importance
- Run k-prototypes and/or k-modes clustering clustering algorithms
- Select the proper number of clusters based on silhouette scoreing
- Explore visualization and feature importance methods for categorical datasets (e.g. Multiple Correspondence Analysis)
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
