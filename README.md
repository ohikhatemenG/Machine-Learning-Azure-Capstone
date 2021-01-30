# Machine-Learning-Azure-Capstone
This is a repository for Machine Learning Engineer for Azure Nanodegree
# The main Objective of this Project
The main objective of this project is to use the Azure Machine Learning Hyper Drive and AutoMl platform and capabilities to build a machine learning model and deploy the best model based based on the performance evaluation metric.
See the Architecture diagram for more insights below
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Architecture%20Diagram%202.png'>
# Business Problem and Statement
Diabetes has being a common diseases affecting the globe in recent time. Many Health Institutions, Health Experts and Goverment have being engaging in research to come up with permanent cure for this but could not be able to do it. So I wanted to build a machine learning model with the  National Institute of Diabetes and Digestive and Kidney Diseases data available and  diagnostically predict whether or not a patient has diabetes.
# Dataset
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Disease
I derived the dataset from Kaggle and the link can be found below.
https://www.kaggle.com/uciml/pima-indians-diabetes-database/diabetes.
# Dataset description
Pregnancy : Number Of Pregnacies happend
Glucose : Blood Glucose Levels
Blood Pressure : Blood Pressure Levels
Skin ThickNess : Triceps Skin Thickness
Insulin : Blood Insulin Levels
BMI :Body Mass Index
Diabetes :Diabetes Function
Age : Age Of Patient in Years
Outcome : 1 0r 0 to indicate whether a patient has diabetes or Not
# Machine Learning Model 
The data has lot of outliers, some have irrelavant values. All the preprocessing was done and placed in train.py using clean data function. After that I choose Logistic Regression Model to build as the Problem is binary problem and choosen accuracy as primary metric which needs to be maximised.
# Registered the Dataset on Azure
Since this is an external dataset I need to register the dataset by going to create a new dataset in Azure Machine learning studio, also I can do this by using python sdk
I have tested both ways. Once Dataset is registered I need to convert the dataset to Tabular dataset using Tabular Dataset factory module of Azure Machine learning.
# Hyperparameter Tuning using HyperDrive
As mentioned above the model I am using is Logistic regression, I have choosen Inverse Regularisation factor (--C) which penalises the model to prevent over fitting and maximum number of iteration(--Max_iter) as other Hyperparameter to be tuned.
The Sample method I have choosen is Random parameter sampling to tune the Hyper parameters to save the computational cost and time of completion.
Early termination policy I have choosen is Bandit policy as it terminates the model building if any of the conditions like slack factor, slack amout, delay interval are not met as per prescribed limits during maximizing the accuracy.
We need to pass the hyperparameter using an entry script in my case its train.py file having all the preprocessing steps and Hyperparameters using arg.parser method to be passed to Hyper Drive.
Once I pass the Hyper parameters using train.py and submit the Job Hyper Drive will create number of jobs based on the parameters given in Hyperdrive configuration using the combinations of Hyper parameters . After running all the 20 models we find the best model and register it in the portal.
See the Hyper Drives running as below.
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/hyperdrive%20a.png'>
This is the Rundetails widget and showing the progress of the training runs of the different experiments



# Screecast Record
https://youtu.be/Lt1yAkETgo8
