import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data_file = pd.read_csv('C:/Users/Taisa/Desktop/tensor/Tensorflow-Bootcamp-master/02-TensorFlow-Basics/pima-indians-diabetes.csv')
print(data_file.head())

print(data_file.columns)

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree', 'Age']

data_file[cols_to_norm] = data_file[cols_to_norm].apply(lambda x: ( x-x.min() )/  (x.max()-x.min() )  )
num_preg = tf.feature_column.numeric_column('Number_pregnant')
gluc_preg = tf.feature_column.numeric_column('Glucose_concentration')
blood_preg = tf.feature_column.numeric_column('Blood_pressure')
trip_preg = tf.feature_column.numeric_column('Triceps')
insul_preg = tf.feature_column.numeric_column('Insulin')
bmi_preg = tf.feature_column.numeric_column('BMI')
pedi_preg = tf.feature_column.numeric_column('Pedigree')
age_preg = tf.feature_column.numeric_column('Age')


assigned_group  = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])

data_file['Age'].hist(bins = 20)

age_bucket = tf.feature_column.bucketized_column(age_preg, boundaries=[20,30,40,50,60,70,80])

feat_columns = [num_preg, gluc_preg, blood_preg, trip_preg, insul_preg, bmi_preg, pedi_preg, assigned_group,  age_bucket]

x_data = data_file.drop ('Class', axis=1)
labels = data_file['Class']

x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True )
model = tf.estimator.LinearClassifier(feature_columns=feat_columns, n_classes=2)
model.train(input_fn=input_func, steps= 1000)

eval_input = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False )
results = model.evaluate(eval_input)

predict_func =  tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1, shuffle=False )
prediction = model.predict(predict_func)
my_pred = list(prediction)


print(my_pred)


dnn_model = tf.estimator.DNNClassifier(hidden_units= [10,10,10], feature_columns= feat_columns, n_classes= 2)


embed_col = tf.feature_column.embedding_column(assigned_group, dimension=4)

ft_col = [num_preg, gluc_preg, blood_preg, trip_preg, insul_preg, bmi_preg, pedi_preg, embed_col, age_bucket]
aaaaaa_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=10, num_epochs=1000, shuffle=True )

dnn_model = tf.estimator.DNNClassifier(hidden_units= [10,10,10], feature_columns= ft_col, n_classes= 2)
dnn_model.train(input_fn=aaaaaa_func, steps= 1000)
eval_input = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False )
dnn_model.evaluate(eval_input)
plt.show()