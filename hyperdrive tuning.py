#!/usr/bin/env python
# coding: utf-8

# In[7]:


from azureml.core import Workspace, Experiment

from azureml.core import Workspace, Experiment
ws = Workspace.from_config()

ws = Workspace.get(name='quick-starts-ws-136238',
                   subscription_id='cdbe0b43-92a0-4715-838a-f2648cc7ad21',
                   resource_group='aml-quickstarts-136238',
                   )

exp = Experiment(ws, 'myexperiment')

run = exp.start_logging()


# In[12]:


from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config() # this automatically looks for a directory .azureml

# Choose a name for your CPU cluster
cpu_cluster_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                            max_nodes=4, 
                                                            idle_seconds_before_scaledown=2400,
                                                            vm_priority='lowpriority')
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
compute_targets = ws.compute_targets
for name, ct in compute_targets.items():
    print(name, ct.type, ct.provisioning_state)


# In[13]:


from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform,choice
import os

# Specify parameter sampler
ps =RandomParameterSampling( {
        "--C": choice(0.25, 0.5,0.75,1.0,2),
        "--max_iter": choice(100,150,200,250,300)
    }
)
# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory='./',compute_target='cpu-cluster',entry_script='./training/train.py')

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(estimator=est,
                             hyperparameter_sampling=ps,
                             policy=policy,
                             primary_metric_name="Accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=20,
                             max_concurrent_runs=4)


# In[14]:


hyperdrive_run = exp.submit(config=hyperdrive_config)


# In[15]:


RunDetails(hyperdrive_run).show()
hyperdrive_run.wait_for_completion(show_output=True)


# In[16]:


best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['arguments']
print('Best Run Id: ', best_run.id)
print('\n Accuracy:', best_run_metrics['Accuracy'])


# In[17]:


parameter_values = best_run.get_details()['runDefinition']['arguments']
print(parameter_values)


# In[18]:


best_model=best_run.register_model(model_name='Bestmodel.joblib',model_path='outputs')
best_model.id


# In[19]:


print('Best Run Id: ', best_run.id)


# In[ ]:




