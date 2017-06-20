### Slide No. 1
Project Goal/Problem:  
[HOMER Energy](http://www.homerenergy.com/) is a world leader in accurately modeling microgrid optimization. The HOMER (Hybrid Optimization of Multiple Energy Resources) software allows users to understand how to build cost effective and reliable microgrids. To provide business intelligence to its vendor partners, HOMER need to better understand who is using its software. The project aims to segment HOMER software users into groups, from which the realness or seriousness of a user can be inferred.

### Slide No. 2
Process: Data Munging  
Steps taken:  
- Consulted with HOMER counterpart to determine which user inputs are considered most important for interpreting user 'realness'
- Bucketed User Role and Organization Type inputs into general categories. For example, undergraduate students, post-graduate students, and faculty were all tagged as 'Academic'.
- Created True/False boolean features for inputs perceived as important. For example, whether or not a user imported her own wind or solar configuration, as opposed to using the generic settings provided by the software.
- Dropped observations that lacked a geographic location (latitude and longitude coordinates)

### Slide No. 3
Process: Dataframe Creation  
Steps taken:
- Created a user dataframe grouped by user ID and calculated the total number of simulations run by each user. The most common user input was used for all other dataframe features.
- A project country was derived using reverse geocoding, and the Federal Information Processing Standards (FIPS) code for each U.S.-based simulation was scrapped from the FCC for future county-by-county visualizations.

### Slide No. 4
Process: Model Development  
Steps taken:
- A variety of unsupervised clustering algorithms capable of handing nominal data were tested on the cleaned dataset. Models tested include: __KModes__, __KPrototypes__, __Agglomerative (Hierarchical) Clustering__, and __Gaussian Mixture Models__.
- KModes was selected due to its direct applicability to categorical clustering.
- A variety of dimensionality reduction techniques were performed to visualize the model's output, however, a faithful 2D representation of the data and clusters was not achieved.

### Slide No. 5
Visualization: Cluster Counts  
![Counts Bar Graph](img/cluster_counts.png)

### Slide No. 6
Visualization: User Role Heat Map   
![User Role Heat Map](img/user_heatmap.png)

### Slide No. 7
Visualization: U.S. County Heat Map  
![County Choropleth Map](img/maps/choro_map.html)
