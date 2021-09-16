



AutoNeuro
Technical Design Document
Version 1.0

RESTRICTED DISTRIBUTION
The information is standard Company Confidential, but due to its sensitivity, it has restricted distribution and viewing within iNeuron.

Document Version Control
Date Issued
Version 
Description
Author
25th June 2020
1.0
Initial Draft
Virat Sagar

Contributors
The content of this document has been authored with the combined input of the following group of key individuals.
Name
Section Worked Upon
Virat Sagar
Initial Draft

Document Classification
Classification
Company Confidential
Definition
Information is Group confidential and needs to be protected
Context
Where the loss of information confidentiality would result in significant harm to the interests of the organisation, financial loss, embarrassment or loss of information




Contents
1	Technical Design Document	2
1.	Introduction	5
High level objectives	5
2	Workflow Overall	6
Application Flow	6
Exception Scenarios  Overall	6
3	Workflow Data Ingestion and File Conversion	8
3.1	Technical solution design	10
3.2	Exceptions Scenarios	10
4	Stats Based EDA	11
4.1	Steps	11
4.2	Technical solution design	11
4.3	Exceptions Scenarios Module Wise	11
5	Graph-Based EDA	12
5.1	Technical solution design	12
5.2	Exceptions Scenarios Module Wise	13
6	Library Based Utils	14
6.1	Technical solution design	14
6.2	Exceptions Scenarios Module Wise	14
7	Data Transformers( Pre-processing steps)	15
7.1	Technical solution design	15
7.2	Exceptions Scenarios Module Wise	15
8	ML Model Selection	16
8.1	Technical solution design	16
8.2	Exceptions Scenarios Module Wise	16
9	Model Tuning and Optimization	17
9.1	Technical solution design	18
9.2	Exceptions Scenarios Module Wise	18
10	Testing Modules	19
10.1	Technical solution design	19
10.2	Exceptions Scenarios Module Wise	20
11	Prediction Pipeline	21
11.1	Technical solution design	22
11.2	Exceptions Scenarios Module Wise	23
12	Deployment Strategy	24
12.1	Technical solution design	25
12.2	Exceptions Scenarios Module Wise	26
13	Monitoring	27
13.1	Technical solution design	27
13.2	Exceptions Scenarios Module Wise	27
14	Logging	28
14.1	Technical solution design	28
14.2	Exceptions Scenarios Module Wise	28
15	Hardware Requirements	29
15.1	Requirements for model training	29
15.2	Requirements for model testing	29


Introduction
The goal here is to build an end to end automated Machine Learning solution where the user will only give the data and select the type of problem, and the result will be the best performing hyper tuned Machine Learning model. The user will also get privileges to choose the deployment options.
This project shall be delivered in two phases:
Phase 1: All the functionalities with PyPi packages.
Phase2: Integration of UI to all the functionalities. 
The technical design document gives a design blueprint of the Autoneuro project. This document communicates the technical details of the solution proposed.
In addition, this document also captures the different workflows involved to build the solution, exceptions in the workflows and any assumptions that have been considered. 
Once agreed as the basis for the building of the project, the flowchart and assumptions will be used as a platform from which the solution will be designed.
Changes to this business process may constitute a request for change and will be subject to the agreed agility program change procedures.
Note: All the code will be written in python version 3.7

High level objectives
The high-level objectives are:
Enable reading/loading of data from the various sources and convert them into pandas dataframe(details mentioned in the Data Ingestion Section).
Enable reading various file formats and convert them into pandas dataframe(details mentioned in the Data Ingestion Section).
Give user the option to specify feature and target columns.
Give user the option to select the problem type, viz. Regression, Classification (include anomaly detection), Clustering or Time Series. 
Perform statistical analytics of the data and prepare a table for the analysis and show it on screen.
Perform graphical analysis for the data and Showcase the results (graphs) on the screen.
Perform data cleaning operation with all the steps required and showcase a report on screen.
 After data cleaning showcase the graphical analysis once again for comparison.
Check whether clustering is required or not.
Choose the appropriate ML model for training.
Perform model Tuning.
Create a list of top 3 models  and show multiple metrics for them.
Give option for prediction.
Give options for docker container creation.
Give option for automatic cloud deployment.
Phase 1: Create Pypi packages
Phase 2: Create UI
Workflow Overall
Application Flow



Exception Scenarios  Overall
Step
Exception
Mitigation
User gives Wrong Data Source
Give proper error message
Ask the user to re-enter the details
User gives corrupted data 
Give proper error message


User gives wrong null symbol
Give proper error message
Ask the user to provide correct symbol used for missing values
If the cluster contains only one class
No error message required
Handle this exception internally. User doesn’t know.
Deployment credentials are wrong
Give proper error message
Ask for the details to be entered again

Workflow Data Ingestion and File Conversion
Data Sources:
Phase 1:
Data Connector Utils
File Conversion Utils
Microsoft Access
CSV & text files, PDF
Spatial File
JSON
Statistical File
HTML
Tableau Server or Tableau Online
Excel files
Actian Matrix
OpenDocument Spreadsheets
Actian Vectorwise
Binary Excel (.xlsb) files
Alibaba AnalyticDB for MySQL
Clipboard
Alibaba Data Lake Analytics
Pickling
Alibaba MaxCompute
msgpack
Amazon Athena
HDF5 (PyTables)
Amazon Aurora for MySQL
Feather
Amazon EMR Hadoop Hive
Parquet
Amazon Redshift
ORC
Anaplan
Google BigQuery
Apache Drill
Stata format
Aster Database
SAS formats
Azure SQL Synapse Analytics
SPSS formats
Box
Other file formats
Cloudera Hadoop
Performance considerations
Databricks


Denodo


Dropbox


Esri ArcGIS Server


Exasol


Firebird 3


Google Ads


Google Analytics


Google BigQuery


Google Cloud SQL


Google Drive


Google Sheets


Hortonworks Hadoop Hive


IBM BigInsights


IBM DB2


IBM PDA (Netezza)


Impala


Intuit QuickBooks Online


Kognitio


Kyvos


LinkedIn Sales Navigator


MapR Hadoop Hive


MariaDB


Marketo


MarkLogic


MemSQL


Microsoft Analysis Services


Microsoft PowerPivot


Microsoft SQL Server


MonetDB


MongoDB BI Connector


MySQL


OData


OneDrive


Oracle


Oracle Eloqua


Oracle Essbase


Pivotal Greenplum


PostgreSQL


Presto


Progress OpenEdge


Qubole Presto


Salesforce


Splunk


SAP HANA


SAP NetWeaver Business Warehouse


SAP Sybase ASE


SAP Sybase IQ


ServiceNow ITSM


SharePoint Lists


Snowflake


Spark SQL


















Connector Plugin


Web Data Connector


Other Databases (JDBC)


Other Databases (ODBC)




Phase 2:
Data Connector Utils
File Conversion Utils
Spatial File
OpenDocument Spreadsheets
Statistical File


Tableau Server or Tableau Online


Actian Matrix


Teradata OLAP Connector


TIBCO Data Virtualization (Cisco Information Server)


Vertica


Teradata




Technical solution design
                  
Method Definitions
Class Name

DataGetter


Method Name
read_data_from_csv




Method Description
This method will be used to read data from a csv file or a flat file


Input parameter  names
self,file_name, header,names, use_cols, separator


Input Parameter Description
file_name: name of the file to be read
header: Row number(s) to be used as column names
names : array-like, optional
    List of column names to use. If file contains no header row, then you
    should explicitly pass ``header=None``.
Use_cols:  To load a subset of columns
Separator: Delimiter to use


ouptput
A pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
read_data_from_json




Method Description
This method will be used to read data from a json file.



Input parameter  names
self,file_name


Input Parameter Description
file_name: name of the file to be read




ouptput
A pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
read_data_from_html




Method Description
This method will be used to read data from an HTML web page


Input parameter  names
self,url


Input Parameter Description
url: URL of the HTML page to be read. 




ouptput
A pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
read_data_from_excel




Method Description
This method will be used to read data from an MS Excel File


Input parameter  names
self,file_name,sheet_name, header,names, use_cols, separator


Input Parameter Description
file_name: name of the file to be read
sheet_name: Lists of strings/integers are used to request
    multiple sheets. Specify None to get all sheets.
header: Row number(s) to be used as column names
names : array-like, optional
    List of column names to use. If file contains no header row, then you
    should explicitly pass ``header=None``.
Use_cols:  To load a subset of columns
Separator: Delimiter to use


ouptput
A pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
Connect_to_sqldb




Method Description
This method will be used to connect to a SQL Databases


Input parameter  names
self,host,port, username, password


Input Parameter Description
host: the server hostname/IP where the DB server is hosted
Port: the port at which the DB Server is running
username: The username to connect to the DB server
password: The password to connect to the DB server




ouptput
A DB connection object


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
read_data_from_sqldb




Method Description
This method will be used to read data from SQL Databases


Input parameter  names
self,db_name,host,port, username, password, schema_name,query_string


Input Parameter Description
db_name: For example, SQL, MySQL, SQLLite etc.
host: the server hostname/IP where the DB server is hosted
Port: the port at which the DB Server is running
username: The username to connect to the DB server
password: The password to connect to the DB server
schema_name: The name of the DB schema the user wants to connect to.
query_string: the query to be executed to load the data


ouptput
A Pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
read_data_from_mongdb




Method Description
This method will be used to read data from Mongo DB


Input parameter  names
self,host,port, username, password, db_name,collection_name, query_string
‘
Input Parameter Description
host: the server hostname/IP where the DB server is hosted
Port: the port at which the DB Server is running
username: The username to connect to the DB server
password: The password to connect to the DB server
db_name: The name of the database
collection_name: The name of the collection the user wants to connect to.
query_string: the query to be executed to load the data


ouptput
A Pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message



Exceptions Scenarios 

Step
Exception
Mitigation
User gives Wrong Data Source
Give proper error message
Ask the user to re-enter the details
User gives corrupted data 
Give proper error message




Data Profiling
After reading the data, automatically the following details should be shown:
The number of rows
The number of columns
Number of missing values per column and their percentage
Total missing values and it’s percentage
Number of categorical columns and their list
Number of numerical columns and their list
Number of duplicate rows
Number of columns with zero standard deviation and their list
Size occupied in RAM

Method Definition

Class Name

DataProfiler


Method Name
get_data_profile




Method Description
This method will be used to give various insighst about data.


Input parameter  names
self, dataframe


Input Parameter Description
dataframe: the inpt data just loaded from source 


ouptput
The number of rows
The number of columns
Number of missing values per column and their percentage
Total missing values and it’s percentage
Number of categorical columns and their list
Number of numerical columns and their list
Number of duplicate rows
Number of columns with zero standard deviation and their list
Size occupied in RAM




On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message


Stats Based EDA
Steps
MVP
OLS
VIF
Correlation
Phase1:
Column contributions/ importance
Annova Test
Chi Square test
Z test
T -test
Weight of Evidence 
F – Test
Phase 2:
Seasonality
Stationary Data


Technical solution design

           
Method Definitions
Class Name

SatisticalDataAnalyser


Method Name
get_correlation




Method Description
This method will be used to get correlation coefficient across all variables in a dataset and remove variables with correlation coefficient value greater than 0.60 (by default)


Input parameter  names
self, dataframe, threshold


Input Parameter Description
dataframe: the input data loaded from the source
threshold: threshold value for removing highly correlated variables. By default, use 0.60


ouptput
Multicollinearity free pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
get _ols_summary




Method Description
This method will be used to get the OLS summary of the dataset. The variables having lower p-value will be kept and others will be dropped



Input parameter  names
self, dataframe


Input Parameter Description
dataframe: the input data loaded from the source


ouptput
OLS Summary


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
get_vif_report




Method Description
This method will be used to get the VIF report of the dataset


Input parameter  names
self, dataframe, target variable


Input Parameter Description
dataframe: the input data loaded from the source
target variable: target variable of the dataset which will be excluded while calculating VIF of the dataset


ouptput
VIF report


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message


Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Column has mixed values(Integer & number)
Give proper error message
Ask the user to correct the data.
Not all values are numbers 
Handle Internally
Convert categorical to numerical values


Graph-Based EDA
Create the following graphs:
MVP:
Correlation Heatmaps
Check for balance/imbalance
Phase1:
Count plots
Boxplot for outliers
Piecharts for categories
Geographical plots for scenarios
Line charts for  trends
Barplots
Area Charts
KDE Plots
Stacked charts
Scatterplot
Phase 2:
Word maps
PACF
ACF
Add Custom controls sliders etc

Note: We are going to use plotly for all the graphs.( https://plotly.com/python/)

Technical solution design
 	   
Method Definitions
Class Name

DataVisualization(Dummy)


Method Name
read_data_from_csv




Method Description
This method will be used to read data from a csv file or a flat file


Input parameter  names
self,file_name, header,names, use_cols, separator


Input Parameter Description
file_name: name of the file to be read
header: Row number(s) to be used as column names
names : array-like, optional
    List of column names to use. If file contains no header row, then you
    should explicitly pass ``header=None``.
Use_cols:  To load a subset of columns
Separator: Delimiter to use


ouptput
A pandas Dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message


Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Wrong input to the methods 
Handle Internally
Code should never give a wrong input


Data Transformers( Pre-processing steps)
MVP:
Null value handling
Categorical to numerical
Imbalanced data set handling
Handling columns with std deviation zero or below a threshold
Normalisation
PCA
Phase1:
Outlier detection
Data Scaling/ Normalisation
Feature Selection: https://scikit-learn.org/stable/auto_examples/index.html#feature-selection


Technical solution design

Method Definitions
Class Name

DataPreprocessor


Method Name
impute_missing_values




Method Description
This method will be used to impute missing values in the dataframe


Input Parameter Names
self, data, strategy, impute_val, missing_vals, mv_flag


Input Parameter Description
data : name of the input dataframe
strategy : strategy to be used for MVI (Missing Value Imputation)
  --‘median’ : default for continuous variables, replaces missing value(s) with median of the concerned column
  --‘mean’
  --‘mode’ : default for categorical variables
  --‘fixed’ : replaces all missing values with a fixed ‘explicitly specified’ value
impute_val : None(default), can be assigned a value to be used for imputation in ‘fixed’ strategy
missing_vals : None(default), a list/tuple of missing value indicators. By default, it considers only NaN as missing. Dictionary can be passed to consider different missing values for different columns in format – {col_name:[val1,val2, …], col2: […]}
mv_flag : None(default), can be passed list/tuple of columns as input for which it creates missing value flags


output
A DataFrame with missing values imputed


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
type_conversion




Method Description
This method will be used to convert column datatype from numerical to categorical or vice-versa, if possible.


Input Parameter Names
self, dataset, cat_to_num, num_to_cat


Input Parameter Description
dataset : input DataFrame in which type conversion is needed
cat_to_num : None(default), list/tuple of variables that need to be converted from categorical to numerical
num_to_cat : None(default), list/tuple of variables to be converted from numerical to categorical


output
A DataFrame with column types changed as per requirement


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
remove_imbalance




Method Description
This method will be used to handle unbalanced datasets(rare classes) through oversampling/ undersampling techniques


Input Parameter Names
self, data, threshold


Input Parameter Description
data: the input dataframe with target column.
threshold: the threshold of mismatch between the target values to perform balancing.


output
A balanced dataframe


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
remove_columns_with_minimal_variance




Method Description
This method drops any numerical column with standard deviation below specified threshold


Input Parameter Names
self, data, threshold


Input Parameter Description
data: input DataFrame in which we need to check std deviations
threshold : the threshold for std deviation below which we need to drop the columns 


output
A DataFrame with numerical columns with low std dev dropped


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
normalize_data




Method Description
This method will be used to do a standardization, normalisation, min-max scaling of numerical variables the input DataFrame


Input Parameter Names
self, data, strategy, mean, std


Input Parameter Description
data : input DataFrame in which transformation is to be applied
strategy : transformation to be used on the numerical columns
  -- ‘normal’ : transforms data to std. normal distribution with mean=0 and std=1.
  -- ‘standardize’ : standardizes data using mean and std specified
  -- ‘minmax’ : does a min-max scaling for numerical columns
mean : 0(default), mean around which standardisation needs to be dome
std : 1(default), standard deviation that needs to be applied for transformation

*further mathematical transformations(for instance:log, inverse) can also be included in strategy and an additional function parameter to take input function.


output
A DataFrame with all the numerical columns transformed as per requirement


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
pca






Method Description
This method will be used to do the Principal Component Analysis on input dataframe and select the most important components


Input Parameter Names
self, data, var_explained


Input Parameter Description
data :  input DataFrame in which pca is to be applied
var_explained : 0.90(default), Total variation(0 to 1) that we want the selected variables to be able to explain


output
A DataFrame with original variables and its principal components.


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
get_categorical_encoding






Method Description
This method does categorical encoding and is largely dependent based on this below package:
https://pypi.org/project/category-encoders/


Input Parameter Names
Will depend on how this package is being used


output
A DataFrame with encoded features and the original categorical columns both. Original categorical columns can be dropped, if perceived necessary


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message


Exceptions Scenarios Module Wise

Step
Exception
Mitigation
Wrong parameters passed to the methods 
Handle Internally
Code should never give a wrong input


ML Model Selection
MVP:
3 Models—KNN, RandomForest, XGBoost
Phase1:
Model Selection criteria
Technical solution design

Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Wrong parameters passed to the methods 
Handle Internally
Code should never give a wrong input


Model Tuning and Optimization
Note: The data should have been divided into train and validation set before this.
Methods for hyper tuning all kinds of models.
Regression:
Linear Regression
Decision Tree
Random Forest
XG Boost
Support Vector Regressor
KNN Regressor

Model selection criteria:
MSE, RMSE, R squared, adjusted R squared
Classification:
Logistic Regression
Decision Tree
Random Forest
XG Boost
Support Vector Classifier
KNN Classifier
Naïve Baye’s

Model selection criteria:
Accuracy, AUC, Precision, Recall, F Beta

Clustering:
K-Means
Hierarchial
DBSCAN
Phase 2:
GLM
GAM (https://www.statsmodels.org/stable/regression.html)
Time Series
Anomaly Detection
Novelty Detection
Optics
Gaussian Mixtures
BIRCH
NLP
Deep Learning
Regularization modules if necessary


Technical solution design
 

Method Definitions
Class Name

ModelTuner


Method Name
get_tuned_knn_model




Method Description
This method will be used to get the hypertuned KNN Model


Input parameter  names
self,data


Input Parameter Description
Data: the training data


Hyperparameters to tune
n_neighbors:Number of neighbors to use by default for kneighbors queries.
weights: weight function used in prediction. Possible values:
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Algorithm used to compute the nearest neighbors:
leaf_size: int, default=30
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
n_jobs: int, Keep it as -1


ouptput
A hyper parameter tuned model object


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
get_tuned_random_forest_model




Method Description
This method will be used to get the hypertuned Random Forest Model


Input parameter  names
self,data


Input Parameter Description
Data: the training data


Hyperparameters to tune
Classifier🡪 
n_estimators: The number of trees in the forest.
criterion{“gini”, “entropy”}, default=”gini”
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 
max_depth: int, default=None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split: int or float, default=2
The minimum number of samples required to split an internal node:
n_jobs= -1

Regressor🡪 

n_estimators: The number of trees in the forest.
criterion{“mse”, “mae”}, default=”mse”
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 
max_depth: int, default=None
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split: int or float, default=2
The minimum number of samples required to split an internal node:





ouptput
A hyper parameter tuned model object


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message
Method Name
get_tuned_xgboost_model




Method Description
This method will be used to get the hypertuned XGBoost Model


Input parameter  names
self,data


Input Parameter Description
Data: the training data


Hyperparameters to tune
eta [default=0.3, alias: learning_rate]
Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
range: [0,1]
gamma [default=0, alias: min_split_loss]
Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
range: [0,∞]
max_depth [default=6]
Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguided growing policy when tree_method is set as hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
Objective: The objective function


ouptput
A hyper parameter tuned model object


On Exception
Write the exception in the log file.
Raise an exception with the appropriate error message


Exceptions Scenarios Module Wise
Step
Exception
Mitigation








Testing Modules
Divide the training data itself into  train and test sets
Use test data to have tests run on the three best models
Give the test report
R2 Score
Adjusted R2 score
MSE
Accuracy
Precision
Recall
F Beta
Cluster Purity
Silhouette score 

Phase 2
AIC
BIC

Note: Save the best model after validation is completed.
Technical solution design

Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Number of Parameters do not match
Handle internally
Check the test data creation and verify the  columns
Only once class present in test data
Handle Internally




Prediction Pipeline  
Use the existing data read modules
Use the existing pre-processing module
Load the model into memory
Do predictions
Store  prediction results(show sample predictions)
Phase 2:
UI for predictions

Technical solution design

Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Columns don’t match in training and Prediction data
Show error message
The user enters the correct data








Deployment Strategy 
Take the cloud name as input
Prepare the metadata files based on cloud
Phase 2:
Accept the user credentials
Prepare a script file to push changes
Docker instance
Push of the docker instance to cloud

Technical solution design

Exceptions Scenarios Module Wise
Step
Exception
Mitigation
Wrong Cloud credentials
Show error message
The user enters the correct data
Docker instance not working
Show error message
Fix the error
Cloud push failed
Show the error
Make corrections to the metadata 
files
Cloud app not starting


Ask the user for cloud logs for debugging


Monitoring
Phase 2
No. Of predictions for individual classes
No. of  predictions (per day, per hour, per week etc.)
No. of hits
Training data size (number of rows)
Time spent in training
Failures


Technical solution design

Exceptions Scenarios Module Wise
Step
Exception
Mitigation








Logging
Separate Folder for logs
Logging of every step
Entry to the methods
Exit from the methods with success/ failure message
Error message Logging
Model comparisons
Training start and end
Prediction start and end
Achieve asynchronous logging

Phase 2:
Options for Logging in DB
Options for Log Publish



Technical solution design


 Common Logging Framework Code
Class Name

App Logger
Method Name
log
Method Description
This method will be used for logging all the information to the file.
Input parameter  names
self,file_object, log_message
Input Parameter Description
file_object: the file where the logs will be written
log_message: the message to be logged
ouptput
A log file with messages

from datetime import datetime
class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):“””This method will be used for logging all the information to the file.”””
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")

Exceptions Scenarios Module Wise

Ideally, the logging should never fail.

Hardware Requirements
Requirements for model training
The minimum configuration should be:
8 GB RAM
2 GB of Hard Disk Space
Intel Core i5 Processor
Requirements for model testing
The minimum configuration should be:
4 GB RAM
2 GB of Hard Disk Space
Intel Core i5 Processor

Sample code and standard to be followed:
Sample Code:
class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """
    def __init__(self, file_object, logger_object):
         self.training_file='Training_FileFromDB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
	Input Description: 	
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,'Entered the get_data method of the Data_Getter class') # Logging entry to the method
        try:
            self.data= pd.read_csv(self.training_file) # reading the data file
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class') # Logging exit from the method
            return self.data # return the read data to the calling method
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e)) # Logging the exception message
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class') # Logging unsuccessful load of data
            raise Exception() # raising exception and exiting

Coding Standard:
 Imports should usually be on separate lines
Avoid trailing whitespace anywhere. Because it's usually invisible, it can be confusing.
Compound statements (multiple statements on the same line) are generally discouraged
Comments should be complete sentences. Always make a priority of keeping the comments up-to-date when the code changes. Ensure that your comments are clear and easily understandable to other speakers of the language you are writing in.
Never use the characters 'l' (lowercase letter el), 'O' (uppercase letter oh), or 'I' (uppercase letter eye) as single character variable names.
The name of the variables should start with small case capital letters and a multi word variable should be named as: word1_word2_word3.
The variable name should be appropriate based on the things that they do. DO NOT USE NAMES LIKE x, k, y etc.  Always use a meaningful English word. For example, customer_name, nearest_neighbour etc.
Method names should start with small case characters. They should start with a verb and make a meaningful sense of what they are supposed to accomplish. For e.g.: load_data_from_sql()
Always use self for the first argument to instance methods.
Class names should normally use the CapWords convention. Class name should also represent the functionality of the class. For e.g. DataLoader()
Modules/Packages/Folders should have short, all-lowercase names. Underscores can be used in the module name if it improves readability. For e.g.: data_ingestion
Constants are usually defined on a module level and written in all capital letters with underscores separating words. Examples include MAX_OVERFLOW and TOTAL.
Comparisons to singletons like None should always be done with is or is not, never the equality operators
The code should be properly enclosed withing try and exception blocks and the exceptions should be handled with proper error messages.
Additionally, for all try/except clauses, limit the try clause to the absolute minimum amount of code necessary. Again, this avoids masking bugs
When a resource is local to a particular section of code, use a with statement to ensure it is cleaned up promptly and reliably after use.
Be consistent in return statements. Either all return statements in a function should return an expression, or none of them should. If any return statement returns an expression, any return statements where no value is returned should explicitly state this as return None, and an explicit return statement should be present at the end of the function (if reachable)
Object type comparisons should always use isinstance() instead of comparing types directly
Don't compare boolean values to True or False using ==






