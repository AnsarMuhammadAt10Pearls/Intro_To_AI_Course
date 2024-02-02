# https://www.kaggle.com/code/ansarmuhammad72/house-prices-prediction-using-tfdf/edit


# %% [markdown] {"id":"5v5mm4amQRrm","papermill":{"duration":0.010092,"end_time":"2023-03-07T06:21:39.774967","exception":false,"start_time":"2023-03-07T06:21:39.764875","status":"completed"},"tags":[]}
# # House Prices Prediction using TensorFlow Decision Forests

# %% [markdown] {"id":"Z4eo3rH_MKbC","papermill":{"duration":0.00862,"end_time":"2023-03-07T06:21:39.792607","exception":false,"start_time":"2023-03-07T06:21:39.783987","status":"completed"},"tags":[]}
# This notebook walks you through how to train a baseline Random Forest model using TensorFlow Decision Forests on the House Prices dataset made available for this competition.
#
# Roughly, the code will look as follows:
#
# ```
# import tensorflow_decision_forests as tfdf
# import pandas as pd
#
# dataset = pd.read_csv("project/dataset.csv")
# tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")
#
# model = tfdf.keras.RandomForestModel()
# model.fit(tf_dataset)
#
# print(model.summary())
# ```
#
# Decision Forests are a family of tree-based models including Random Forests and Gradient Boosted Trees. They are the best place to start when working with tabular data, and will often outperform (or provide a strong baseline) before you begin experimenting with neural networks.

# %% [markdown] {"id":"FVOXAyXl3-fA","papermill":{"duration":0.008317,"end_time":"2023-03-07T06:21:39.809564","exception":false,"start_time":"2023-03-07T06:21:39.801247","status":"completed"},"tags":[]}
# ## Import the library

# %% [code] {"id":"IGmyjJJatzBZ","papermill":{"duration":8.300496,"end_time":"2023-03-07T06:21:48.118668","exception":false,"start_time":"2023-03-07T06:21:39.818172","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:55:55.619648Z","iopub.execute_input":"2024-02-01T06:55:55.620843Z","iopub.status.idle":"2024-02-01T06:56:06.711815Z","shell.execute_reply.started":"2024-02-01T06:55:55.620780Z","shell.execute_reply":"2024-02-01T06:56:06.710364Z"}}
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
# matplotlib inline

# %% [code] {"id":"dh4qwB4iN7Ue","papermill":{"duration":0.019012,"end_time":"2023-03-07T06:21:48.149058","exception":false,"start_time":"2023-03-07T06:21:48.130046","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:06.714062Z","iopub.execute_input":"2024-02-01T06:56:06.714750Z","iopub.status.idle":"2024-02-01T06:56:06.722528Z","shell.execute_reply.started":"2024-02-01T06:56:06.714694Z","shell.execute_reply":"2024-02-01T06:56:06.721027Z"}}
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# %% [markdown] {"id":"-3vxMmCPvqpf","papermill":{"duration":0.009922,"end_time":"2023-03-07T06:21:48.16945","exception":false,"start_time":"2023-03-07T06:21:48.159528","status":"completed"},"tags":[]}
# ## Load the dataset
#

# %% [code] {"id":"JVMPH_IDOBH2","papermill":{"duration":0.066785,"end_time":"2023-03-07T06:21:48.245226","exception":false,"start_time":"2023-03-07T06:21:48.178441","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:06.724273Z","iopub.execute_input":"2024-02-01T06:56:06.724788Z","iopub.status.idle":"2024-02-01T06:56:06.778983Z","shell.execute_reply.started":"2024-02-01T06:56:06.724719Z","shell.execute_reply":"2024-02-01T06:56:06.777610Z"}}
train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

# %% [markdown] {"papermill":{"duration":0.008651,"end_time":"2023-03-07T06:21:48.263024","exception":false,"start_time":"2023-03-07T06:21:48.254373","status":"completed"},"tags":[],"id":"mTnx8h9i416m"}
# The data is composed of 81 columns and 1460 entries. We can see all 81 dimensions of our dataset by printing out the first 3 entries using the following code:

# %% [code] {"papermill":{"duration":0.049873,"end_time":"2023-03-07T06:21:48.321938","exception":false,"start_time":"2023-03-07T06:21:48.272065","status":"completed"},"tags":[],"id":"kgbP5R6X416m","execution":{"iopub.status.busy":"2024-02-01T06:56:06.781782Z","iopub.execute_input":"2024-02-01T06:56:06.782191Z","iopub.status.idle":"2024-02-01T06:56:06.830585Z","shell.execute_reply.started":"2024-02-01T06:56:06.782154Z","shell.execute_reply":"2024-02-01T06:56:06.829407Z"}}
dataset_df.head(3)

# %% [markdown] {"papermill":{"duration":0.009123,"end_time":"2023-03-07T06:21:48.340722","exception":false,"start_time":"2023-03-07T06:21:48.331599","status":"completed"},"tags":[],"id":"ulu8XdxO416n"}
# * There are 79 feature columns. Using these features your model has to predict the house sale price indicated by the label column named `SalePrice`.

# %% [markdown] {"papermill":{"duration":0.009025,"end_time":"2023-03-07T06:21:48.359367","exception":false,"start_time":"2023-03-07T06:21:48.350342","status":"completed"},"tags":[],"id":"n82wWtvL416n"}
# We will drop the `Id` column as it is not necessary for model training.

# %% [code] {"papermill":{"duration":0.043419,"end_time":"2023-03-07T06:21:48.412206","exception":false,"start_time":"2023-03-07T06:21:48.368787","status":"completed"},"tags":[],"id":"0lItmbYS416n","execution":{"iopub.status.busy":"2024-02-01T06:56:06.831953Z","iopub.execute_input":"2024-02-01T06:56:06.832500Z","iopub.status.idle":"2024-02-01T06:56:06.866543Z","shell.execute_reply.started":"2024-02-01T06:56:06.832429Z","shell.execute_reply":"2024-02-01T06:56:06.865220Z"}}
dataset_df = dataset_df.drop('Id', axis=1)
dataset_df.head(3)

# %% [markdown] {"papermill":{"duration":0.009883,"end_time":"2023-03-07T06:21:48.432601","exception":false,"start_time":"2023-03-07T06:21:48.422718","status":"completed"},"tags":[],"id":"QA_v408l416n"}
# We can inspect the types of feature columns using the following code:

# %% [code] {"papermill":{"duration":0.046783,"end_time":"2023-03-07T06:21:48.489619","exception":false,"start_time":"2023-03-07T06:21:48.442836","status":"completed"},"tags":[],"id":"du6DU4Of416n","execution":{"iopub.status.busy":"2024-02-01T06:56:06.867999Z","iopub.execute_input":"2024-02-01T06:56:06.868351Z","iopub.status.idle":"2024-02-01T06:56:06.908624Z","shell.execute_reply.started":"2024-02-01T06:56:06.868316Z","shell.execute_reply":"2024-02-01T06:56:06.907278Z"}}
dataset_df.info()

# %% [markdown] {"papermill":{"duration":0.010252,"end_time":"2023-03-07T06:21:48.510224","exception":false,"start_time":"2023-03-07T06:21:48.499972","status":"completed"},"tags":[],"id":"PxdZCHvk416o"}
# ## House Price Distribution
#
# Now let us take a look at how the house prices are distributed.

# %% [code] {"papermill":{"duration":0.497946,"end_time":"2023-03-07T06:21:49.018361","exception":false,"start_time":"2023-03-07T06:21:48.520415","status":"completed"},"tags":[],"id":"qROZWZyE416o","execution":{"iopub.status.busy":"2024-02-01T06:56:06.910283Z","iopub.execute_input":"2024-02-01T06:56:06.910674Z","iopub.status.idle":"2024-02-01T06:56:07.660808Z","shell.execute_reply.started":"2024-02-01T06:56:06.910637Z","shell.execute_reply":"2024-02-01T06:56:07.659773Z"}}
print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});

# %% [markdown] {"papermill":{"duration":0.01022,"end_time":"2023-03-07T06:21:49.039644","exception":false,"start_time":"2023-03-07T06:21:49.029424","status":"completed"},"tags":[],"id":"tKnn1nR-416o"}
# ## Numerical data distribution
#
# We will now take a look at how the numerical features are distributed. In order to do this, let us first list all the types of data from our dataset and select only the numerical ones.

# %% [code] {"papermill":{"duration":0.022381,"end_time":"2023-03-07T06:21:49.0727","exception":false,"start_time":"2023-03-07T06:21:49.050319","status":"completed"},"tags":[],"id":"-hrMItSC416o","execution":{"iopub.status.busy":"2024-02-01T06:56:07.662346Z","iopub.execute_input":"2024-02-01T06:56:07.663256Z","iopub.status.idle":"2024-02-01T06:56:07.672024Z","shell.execute_reply.started":"2024-02-01T06:56:07.663214Z","shell.execute_reply":"2024-02-01T06:56:07.670680Z"}}
list(set(dataset_df.dtypes.tolist()))

# %% [code] {"papermill":{"duration":0.038307,"end_time":"2023-03-07T06:21:49.122233","exception":false,"start_time":"2023-03-07T06:21:49.083926","status":"completed"},"tags":[],"id":"Vg2PQvfb416o","execution":{"iopub.status.busy":"2024-02-01T06:56:07.673466Z","iopub.execute_input":"2024-02-01T06:56:07.673826Z","iopub.status.idle":"2024-02-01T06:56:07.704579Z","shell.execute_reply.started":"2024-02-01T06:56:07.673792Z","shell.execute_reply":"2024-02-01T06:56:07.703288Z"}}
df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
df_num.head()

# %% [markdown] {"papermill":{"duration":0.0106,"end_time":"2023-03-07T06:21:49.144057","exception":false,"start_time":"2023-03-07T06:21:49.133457","status":"completed"},"tags":[],"id":"MnaH5h8u416o"}
# Now let us plot the distribution for all the numerical features.

# %% [code] {"papermill":{"duration":8.021473,"end_time":"2023-03-07T06:21:57.176534","exception":false,"start_time":"2023-03-07T06:21:49.155061","status":"completed"},"tags":[],"id":"Dj4h_dIw416o","execution":{"iopub.status.busy":"2024-02-01T06:56:07.710498Z","iopub.execute_input":"2024-02-01T06:56:07.710874Z","iopub.status.idle":"2024-02-01T06:56:16.880961Z","shell.execute_reply.started":"2024-02-01T06:56:07.710839Z","shell.execute_reply":"2024-02-01T06:56:16.879608Z"}}
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);

# %% [markdown] {"id":"H4O7QCoh5e2e","papermill":{"duration":0.012325,"end_time":"2023-03-07T06:21:57.202216","exception":false,"start_time":"2023-03-07T06:21:57.189891","status":"completed"},"tags":[]}
# ## Prepare the dataset
#
# This dataset contains a mix of numeric, categorical and missing features. TF-DF supports all these feature types natively, and no preprocessing is required. This is one advantage of tree-based models, making them a great entry point to Tensorflow and ML.

# %% [markdown] {"id":"brbRsBQfSC74","papermill":{"duration":0.012106,"end_time":"2023-03-07T06:21:57.227439","exception":false,"start_time":"2023-03-07T06:21:57.215333","status":"completed"},"tags":[]}
# Now let us split the dataset into training and testing datasets:

# %% [code] {"id":"tsQad0t7SBv2","papermill":{"duration":0.025712,"end_time":"2023-03-07T06:21:57.266147","exception":false,"start_time":"2023-03-07T06:21:57.240435","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:16.883079Z","iopub.execute_input":"2024-02-01T06:56:16.884028Z","iopub.status.idle":"2024-02-01T06:56:16.898645Z","shell.execute_reply.started":"2024-02-01T06:56:16.883965Z","shell.execute_reply":"2024-02-01T06:56:16.897157Z"}}
import numpy as np


def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

# %% [markdown] {"id":"-hNGPbLlSGvp","papermill":{"duration":0.01598,"end_time":"2023-03-07T06:21:57.294832","exception":false,"start_time":"2023-03-07T06:21:57.278852","status":"completed"},"tags":[]}
# There's one more step required before we can train the model. We need to convert the datatset from Pandas format (`pd.DataFrame`) into TensorFlow Datasets format (`tf.data.Dataset`).
#
# [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) is a high performance data loading library which is helpful when training neural networks with accelerators like GPUs and TPUs.

# %% [markdown] {"papermill":{"duration":0.012107,"end_time":"2023-03-07T06:21:57.319528","exception":false,"start_time":"2023-03-07T06:21:57.307421","status":"completed"},"tags":[],"id":"7goqxGx3416p"}
# By default the Random Forest Model is configured to train classification tasks. Since this is a regression problem, we will specify the type of the task (`tfdf.keras.Task.REGRESSION`) as a parameter here.

# %% [code] {"id":"xQgimfirSGQ9","papermill":{"duration":0.26438,"end_time":"2023-03-07T06:21:57.596711","exception":false,"start_time":"2023-03-07T06:21:57.332331","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:16.899998Z","iopub.execute_input":"2024-02-01T06:56:16.900364Z","iopub.status.idle":"2024-02-01T06:56:17.264389Z","shell.execute_reply.started":"2024-02-01T06:56:16.900329Z","shell.execute_reply":"2024-02-01T06:56:17.262935Z"}}
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# %% [markdown] {"id":"IUG4UKUyTNUu","papermill":{"duration":0.012451,"end_time":"2023-03-07T06:21:57.62217","exception":false,"start_time":"2023-03-07T06:21:57.609719","status":"completed"},"tags":[]}
# ## Select a Model
#
# There are several tree-based models for you to choose from.
#
# * RandomForestModel
# * GradientBoostedTreesModel
# * CartModel
# * DistributedGradientBoostedTreesModel
#
# To start, we'll work with a Random Forest. This is the most well-known of the Decision Forest training algorithms.
#
# A Random Forest is a collection of decision trees, each trained independently on a random subset of the training dataset (sampled with replacement). The algorithm is unique in that it is robust to overfitting, and easy to use.

# %% [markdown] {"papermill":{"duration":0.0125,"end_time":"2023-03-07T06:21:57.647596","exception":false,"start_time":"2023-03-07T06:21:57.635096","status":"completed"},"tags":[],"id":"VJSwNUdb416p"}
# We can list the all the available models in TensorFlow Decision Forests using the following code:

# %% [code] {"id":"MFmnkRR_Ui9w","papermill":{"duration":0.024872,"end_time":"2023-03-07T06:21:57.685403","exception":false,"start_time":"2023-03-07T06:21:57.660531","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:17.266241Z","iopub.execute_input":"2024-02-01T06:56:17.266758Z","iopub.status.idle":"2024-02-01T06:56:17.275515Z","shell.execute_reply.started":"2024-02-01T06:56:17.266682Z","shell.execute_reply":"2024-02-01T06:56:17.274223Z"}}
tfdf.keras.get_all_models()

# %% [markdown] {"id":"LiFn716FnMVQ","papermill":{"duration":0.012613,"end_time":"2023-03-07T06:21:57.710894","exception":false,"start_time":"2023-03-07T06:21:57.698281","status":"completed"},"tags":[]}
# ## How can I configure them?
#
# TensorFlow Decision Forests provides good defaults for you (e.g. the top ranking hyperparameters on our benchmarks, slightly modified to run in reasonable time). If you would like to configure the learning algorithm, you will find many options you can explore to get the highest possible accuracy.
#
# You can select a template and/or set parameters as follows:
#
# ```rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)```
#
# Read more [here](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel).

# %% [markdown] {"id":"irxAS91IRVAX","papermill":{"duration":0.012674,"end_time":"2023-03-07T06:21:57.73704","exception":false,"start_time":"2023-03-07T06:21:57.724366","status":"completed"},"tags":[]}
#
#

# %% [markdown] {"id":"AUt4j8fLWRlR","papermill":{"duration":0.012522,"end_time":"2023-03-07T06:21:57.762516","exception":false,"start_time":"2023-03-07T06:21:57.749994","status":"completed"},"tags":[]}
# ## Create a Random Forest
#
# Today, we will use the defaults to create the Random Forest Model while specifiyng the task type as `tfdf.keras.Task.REGRESSION`.

# %% [code] {"id":"O7bqOQMYTRXZ","papermill":{"duration":0.079382,"end_time":"2023-03-07T06:21:57.854964","exception":false,"start_time":"2023-03-07T06:21:57.775582","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:17.277329Z","iopub.execute_input":"2024-02-01T06:56:17.277826Z","iopub.status.idle":"2024-02-01T06:56:17.338458Z","shell.execute_reply.started":"2024-02-01T06:56:17.277763Z","shell.execute_reply":"2024-02-01T06:56:17.337291Z"}}
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])  # Optional, you can use this to include a list of eval metrics

# %% [markdown] {"id":"0CzJ5_sh91Yt","papermill":{"duration":0.013391,"end_time":"2023-03-07T06:21:57.881539","exception":false,"start_time":"2023-03-07T06:21:57.868148","status":"completed"},"tags":[]}
# ## Train the model
#
# We will train the model using a one-liner.
#
# Note: you may see a warning about Autograph. You can safely ignore this, it will be fixed in the next release.

# %% [code] {"id":"Ax6RircN92LW","papermill":{"duration":14.312048,"end_time":"2023-03-07T06:22:12.207321","exception":false,"start_time":"2023-03-07T06:21:57.895273","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:17.339865Z","iopub.execute_input":"2024-02-01T06:56:17.340347Z","iopub.status.idle":"2024-02-01T06:56:33.004644Z","shell.execute_reply.started":"2024-02-01T06:56:17.340308Z","shell.execute_reply":"2024-02-01T06:56:33.002118Z"}}
rf.fit(x=train_ds)

# %% [markdown] {"id":"C1HJ6KxRT7IR","papermill":{"duration":0.014187,"end_time":"2023-03-07T06:22:12.236308","exception":false,"start_time":"2023-03-07T06:22:12.222121","status":"completed"},"tags":[]}
# ## Visualize the model
# One benefit of tree-based models is that you can easily visualize them. The default number of trees used in the Random Forests is 300. We can select a tree to display below.

# %% [code] {"id":"mTx73NgET9f8","papermill":{"duration":0.126324,"end_time":"2023-03-07T06:22:12.377534","exception":false,"start_time":"2023-03-07T06:22:12.25121","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:33.006662Z","iopub.execute_input":"2024-02-01T06:56:33.007129Z","iopub.status.idle":"2024-02-01T06:56:33.037964Z","shell.execute_reply.started":"2024-02-01T06:56:33.007091Z","shell.execute_reply":"2024-02-01T06:56:33.036970Z"}}
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# %% [markdown] {"id":"fazbJOgUT1n4","papermill":{"duration":0.015024,"end_time":"2023-03-07T06:22:12.407834","exception":false,"start_time":"2023-03-07T06:22:12.39281","status":"completed"},"tags":[]}
# ## Evaluate the model on the Out of bag (OOB) data and the validation dataset
#
# Before training the dataset we have manually seperated 20% of the dataset for validation named as `valid_ds`.
#
# We can also use Out of bag (OOB) score to validate our RandomForestModel.
# To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm and the rest of the samples are used to finetune the model.The subset of data that is not chosen is known as Out of bag data (OOB).
# OOB score is computed on the OOB data.
#
# Read more about OOB data [here](https://developers.google.com/machine-learning/decision-forests/out-of-bag).
#
# The training logs show the Root Mean Squared Error (RMSE) evaluated on the out-of-bag dataset according to the number of trees in the model. Let us plot this.
#
# Note: Smaller values are better for this hyperparameter.

# %% [code] {"id":"ryddKoqLWrTp","papermill":{"duration":0.229991,"end_time":"2023-03-07T06:22:12.653052","exception":false,"start_time":"2023-03-07T06:22:12.423061","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:33.039223Z","iopub.execute_input":"2024-02-01T06:56:33.039904Z","iopub.status.idle":"2024-02-01T06:56:33.272680Z","shell.execute_reply.started":"2024-02-01T06:56:33.039864Z","shell.execute_reply":"2024-02-01T06:56:33.270393Z"}}
import matplotlib.pyplot as plt

logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

# %% [markdown] {"id":"Y-yMMsK5-3Mr","papermill":{"duration":0.015203,"end_time":"2023-03-07T06:22:12.684147","exception":false,"start_time":"2023-03-07T06:22:12.668944","status":"completed"},"tags":[]}
# We can also see some general stats on the OOB dataset:

# %% [code] {"id":"gdY8DvriTxky","papermill":{"duration":0.032483,"end_time":"2023-03-07T06:22:12.732339","exception":false,"start_time":"2023-03-07T06:22:12.699856","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:33.273797Z","iopub.execute_input":"2024-02-01T06:56:33.274146Z","iopub.status.idle":"2024-02-01T06:56:33.286714Z","shell.execute_reply.started":"2024-02-01T06:56:33.274111Z","shell.execute_reply":"2024-02-01T06:56:33.285366Z"}}
inspector = rf.make_inspector()
inspector.evaluation()

# %% [markdown] {"id":"GAoGJNjg-9sb","papermill":{"duration":0.015817,"end_time":"2023-03-07T06:22:12.764326","exception":false,"start_time":"2023-03-07T06:22:12.748509","status":"completed"},"tags":[]}
# Now, let us run an evaluation using the validation dataset.

# %% [code] {"id":"39x97YqWZlgm","papermill":{"duration":1.513826,"end_time":"2023-03-07T06:22:14.294393","exception":false,"start_time":"2023-03-07T06:22:12.780567","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:33.288634Z","iopub.execute_input":"2024-02-01T06:56:33.289048Z","iopub.status.idle":"2024-02-01T06:56:34.202338Z","shell.execute_reply.started":"2024-02-01T06:56:33.289008Z","shell.execute_reply":"2024-02-01T06:56:34.200945Z"}}
evaluation = rf.evaluate(x=valid_ds, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# %% [markdown] {"id":"LWWqqDLM7WdZ","papermill":{"duration":0.015916,"end_time":"2023-03-07T06:22:14.32683","exception":false,"start_time":"2023-03-07T06:22:14.310914","status":"completed"},"tags":[]}
# ## Variable importances
#
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests.
# Let us list the available `Variable Importances` for Decision Trees:

# %% [code] {"id":"xok16_jMgGZH","papermill":{"duration":0.028662,"end_time":"2023-03-07T06:22:14.371495","exception":false,"start_time":"2023-03-07T06:22:14.342833","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:34.204424Z","iopub.execute_input":"2024-02-01T06:56:34.205313Z","iopub.status.idle":"2024-02-01T06:56:34.215417Z","shell.execute_reply.started":"2024-02-01T06:56:34.205256Z","shell.execute_reply":"2024-02-01T06:56:34.213909Z"}}
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

# %% [markdown] {"id":"USvNgqBR_JR2","papermill":{"duration":0.016135,"end_time":"2023-03-07T06:22:14.404154","exception":false,"start_time":"2023-03-07T06:22:14.388019","status":"completed"},"tags":[]}
# As an example, let us display the important features for the Variable Importance `NUM_AS_ROOT`.
#
# The larger the importance score for `NUM_AS_ROOT`, the more impact it has on the outcome of the model.
#
# By default, the list is sorted from the most important to the least. From the output you can infer that the feature at the top of the list is used as the root node in most number of trees in the random forest than any other feature.

# %% [code] {"id":"eI073gJHgHxr","papermill":{"duration":0.02844,"end_time":"2023-03-07T06:22:14.449021","exception":false,"start_time":"2023-03-07T06:22:14.420581","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-02-01T06:56:34.217580Z","iopub.execute_input":"2024-02-01T06:56:34.218006Z","iopub.status.idle":"2024-02-01T06:56:34.231808Z","shell.execute_reply.started":"2024-02-01T06:56:34.217966Z","shell.execute_reply":"2024-02-01T06:56:34.230391Z"}}
inspector.variable_importances()["NUM_AS_ROOT"]

# %% [markdown] {"id":"qiASD3ei52H6"}
# Plot the variable importances from the inspector using Matplotlib

# %% [code] {"id":"cyyzelTl53AH","execution":{"iopub.status.busy":"2024-02-01T06:56:34.234263Z","iopub.execute_input":"2024-02-01T06:56:34.234705Z","iopub.status.idle":"2024-02-01T06:56:34.610754Z","shell.execute_reply.started":"2024-02-01T06:56:34.234665Z","shell.execute_reply":"2024-02-01T06:56:34.609295Z"}}
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
#
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# TODO: Replace with "plt.bar_label()" when available.
# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()

# %% [markdown] {"papermill":{"duration":0.016075,"end_time":"2023-03-07T06:22:14.482026","exception":false,"start_time":"2023-03-07T06:22:14.465951","status":"completed"},"tags":[],"id":"jM9uB_7T416r"}
# # Submission
# Finally predict on the competition test data using the model.

# %% [code] {"papermill":{"duration":1.717453,"end_time":"2023-03-07T06:22:16.215717","exception":false,"start_time":"2023-03-07T06:22:14.498264","status":"completed"},"tags":[],"id":"gLySv9yJ416s","execution":{"iopub.status.busy":"2024-02-01T06:56:34.612396Z","iopub.execute_input":"2024-02-01T06:56:34.612807Z","iopub.status.idle":"2024-02-01T06:56:35.907630Z","shell.execute_reply.started":"2024-02-01T06:56:34.612767Z","shell.execute_reply":"2024-02-01T06:56:35.906104Z"}}
test_file_path = "test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data,
    task=tfdf.keras.Task.REGRESSION)

preds = rf.predict(test_ds)
output = pd.DataFrame({'Id': ids,
                       'SalePrice': preds.squeeze()})

output.head()




