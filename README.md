# Machine-Learning-Azure-Capstone
This is a repository for Machine Learning Engineer for Azure Nanodegree
# An overview of the Project
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
# Compute Cluster
To run the Jobs I need a cpu cluster to be created before starting the process whcih can be found in both the notebooks .
# Hyperparameter Tuning using HyperDrive
As mentioned above the model I am using is Logistic regression, I have choosen Inverse Regularisation factor (--C) which penalises the model to prevent over fitting and maximum number of iteration(--Max_iter) as other Hyperparameter to be tuned.
The Sample method I have choosen is Random parameter sampling to tune the Hyper parameters to save the computational cost and time of completion.
Early termination policy I have choosen is Bandit policy as it terminates the model building if any of the conditions like slack factor, slack amout, delay interval are not met as per prescribed limits during maximizing the accuracy.
We need to pass the hyperparameter using an entry script in my case its train.py file having all the preprocessing steps and Hyperparameters using arg.parser method to be passed to Hyper Drive.
Once I pass the Hyper parameters using train.py and submit the Job Hyper Drive will create number of jobs based on the parameters given in Hyperdrive configuration using the combinations of Hyper parameters . After running all the 20 models we find the best model and register it in the portal.
See the Hyper Drives running as below.
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/hyperdrive%20a.png'>
This is the Rundetails widget and showing the progress of the training runs of the different experiments
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/hyperdrive%20b.png'>
The best model with it's run id
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/hyperdrive%20c.png'>
The different hyperparameters
# Automl ML
Hyper Drive run needs lot of preprocessing method for the successful building of a model. Azure has AUTOML capalbilities with less preprocessing techniques reuqired . Here we are going to build a automl model for our problem. The Dataste is registered and converted to Tabular Dataset using Tabular dataset Factory module.
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(config).png'>
The AutomL config can be as seen above
The probelm is classification I choose task as classification.
TimeOut is set to 25 minutes sicne the dataset is only 800 rows approximately.
Primary metric is accuracy as we are trying to maximise the accuracy.
label column is the column we are trying to predict here outcome .
Compute target is the cpu-cluster where the computation needs to be done
N_cross_Validations=5 the number of k fold cross validations, since the dataset is small choosen 5
Iterations: Number of iterations to be run 20 , so this checks 20 automl models Max_concurernt_iterations: 5 number of parallel runs at a time, choosing this too high impact performance so choosen 5
Once I write the config file and submit the experiment it starts building models whcih can be seen below.
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(Rundetails).png'>
The Rundetails widget and it's progress of the training runs of the different experiment
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(bestrun).png'>
The best model with it's run id
# compare the model performance
The Runs AutomL gave voting ensemble model as best model with accuracy of 78.65 better than Hyperdrive model which is 74.46.VotingEnsemble model works on taking the majority voting of underlying models and choose the model with highest votes as best model.Hyperdrive model consume more to time to build than Automl model. Hyperdrive model is an indirect process, while Automl model is a direct process of building up a model. Some of the parameters generated by the Automl best model are reg_lambda=0.7292,scale_pos_weight=1,subsample=0.9,verbose=-10 and verbosity
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Best(A).png'>
The best model for deployment
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/model-1.png'>
Parameters generated by Automl best model
# Save and Registered the Best Model
Once I derived the best model, I have to save and registered the best model
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(sava%20%26%20Reg).png'>
Save and Registered the Best Model
# Deployment
Both models cna be deploy, but I choose to deployed automl model because of is better accuarcy compared to HyperDrive Model.
Before Deploying the model, i need to pack all the dependencies into conda environment file whcih are included in the repository. Once I pack the dependencies a docker conatiner is built and pushed to Azure Container isntance.I need to consume the ACI instance using a rest Endpoint. The endpoint deployed will be seen in endpoints section of the Azure Machine learning studio. Before deploying an endpoint we need to define scoring script which defines the entrypoint to the deployment whcih is given in repository.
We need to define inference config and endpoint config which are in jupyter Notebook of Automl.
Once the end point is deployed and showing healthy its ready to cosnume
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(endpoint).png'>
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(endpo).png'>
The endpoint and showing healthy for consume
The Endpoint is consumed using endpoint.py where we use requests library for cosnuming the endpoint.
The sample input to the endpoint is as below
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(request).png'>
From the above I tested two data points and I gone two outcome as expected
In the same way we can test the endpoint on multiple Data points using sample data from the given dataset.
I convert the swagger dataframe into json and pass the json to service endpoint.
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(request2).png'>
<img src='https://github.com/ohikhatemenG/Machine-Learning-Azure-Capstone/blob/main/Automl(output).png'>
This shows the endpoint is functioning successfully.
# Screecast Record
https://youtu.be/Lt1yAkETgo8
# Future Improvement
The model can be converted to ONNX format and deploy on Edge devices.
Applciation insights can be enabled.
Addition of more data will definitly improve the accuracy by eliminating statistical bias
Preventing target leakage
using fewer features


