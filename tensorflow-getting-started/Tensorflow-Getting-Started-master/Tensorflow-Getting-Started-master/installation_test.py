# import tensorflow
import tensorflow as tf
import pandas as pd

from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense ,Dropout,BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
X = pd.DataFrame(california.data, columns = california.feature_names)
y = pd.DataFrame(california.target, columns = california.target_names)

# define base model
def baseline_model():
    # create model
    model = Sequential()
    
    
    
    model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
    
    
    model.add(Dense(8, input_dim=7, activation='relu'))
    
    
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



X_train, X_test, y_train, y_test = train_test_split(
     X.drop(['Latitude','Longitude'], axis = 1),y, test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler
import datetime


scaler =  MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

log_dir = "C:\\Users\\Raushan\\Downloads\\Code\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)

estimator = KerasRegressor(build_fn = baseline_model, epochs=10, batch_size=3, verbose=1,callbacks=[tensorboard_callback])

history=estimator.fit(X_train,y_train)

model = tf.estimator.BoostedTreesRegressor(feature_columns = X.drop(['Latitude','Longitude'], axis = 1).columns, n_batches_per_layer = 3)


feature_columns, n_batches_per_layer, model_dir=None, label_dimension=1, weight_column=None, n_trees=100, max_depth=6, learning_rate=0.1, l1_regularization=0, l2_regularization=0, tree_complexity=0, min_node_weight=0, config=None, center_bias=False, pruning_mode='none', quantile_sketch_epsilon=0.01, train_in_memory=False


%reload_ext tensorboard



%tensorboard dev upload --logdir 'C:\\Users\\Raushan\\Downloads\\Code\\logs\\fit\\20210619-133025\\train'

%tensorboard --logdir=C:/Users/Raushan/Downloads/Code/20210619-161216

tensorboard --logdir 'C:\\Users\\Raushan\\Downloads\\Code\\logs\\fit\\20210619-133025\'
