#!/usr/bin/env python
# coding: utf-8

# # Análisis #1: Invirtiendo Juntos BBVA

# Puntos a considerar en este análisis de datos para estudiar el comportamiento del cliente al invertir
# Obtener la probabilidad de que un cliente sea Inversor y No inversor
# Con una muestra de 30,000 clientes

# Variable de Salida:
# 
# Variables de entrada:

# Cualquier publicación basada en este dataset debe referenciarse como sigue:
# 
# Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
# 
# El dataset original puede encontrarse en el repositorio del UCI Machine Learning Repository.
# 
# 

# # #2: IMPORTAR LIBRERÍAS Y DATASETs

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Necesitaremos montar su disco usando los siguientes comandos:
# Para obtener más información sobre el montaje, puedes consultar: https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory

#from google.colab import drive
#drive.mount('/content/drive')


# In[124]:


# Tienes que incluir el enlace completo al archivo csv que contiene el conjunto de datos
investpeople_df = pd.read_csv('UCI_invest_People.csv')


# In[125]:


investpeople_df


# In[126]:


investpeople_df.describe()
# la media de LIMIT_BAL = 1500, min = 1 y max = 30000
# la media de EDAD = 25 años, mínimo = 21 y máximo = 79
# El promedio de PAY_AMT es de alrededor de 5000


# # #3: VISUALIZAR EL DATASET

# In[35]:


# Veamos si nos faltan datos, ¡afortunadamente NO!
sns.heatmap(investpeople_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[42]:


#Exploracion de cada una de las variables
investpeople_df.hist(bins = 30, figsize = (20,20), color = 'r')


# In[43]:


# Vamos a eliminar la columna con el ID
investpeople_df.drop(['ID'], axis=1, inplace=True)


# In[48]:


investpeople_df


# In[50]:


# ¡Veamos cuántos clientes podrían incumplir con el pago con tarjeta de crédito!
cc_default_df        = investpeople_df[investpeople_df['default.invesment.next.month'] == 1]
cc_nodefault_df      = investpeople_df[investpeople_df['default.invesment.next.month'] == 0]


# In[65]:


# Contamos el número de clientes que se quedaron y se fueron
# Parece que estamos ante un conjunto de datos desbalanceado

print("Total =", len(investpeople_df))

print("Número de clientes que invierten =", len(cc_default_df))
print("Porcentaje de clientes que invierten =", 1.*len(cc_default_df)/len(investpeople_df)*100.0, "%")
 
print("Número de clientes que no invierten =", len(cc_nodefault_df))
print("Porcentaje de clientes no invierten =", 1.*len(cc_nodefault_df)/len(investpeople_df)*100.0, "%")


# In[66]:


# Comparemos la media y el error estándar de los clientes que cumplen y los que no
cc_default_df.describe()


# In[67]:


# Comparemos la media y el error estándar de los clientes que cumplen y los que no
cc_nodefault_df.describe()


# In[68]:


correlations = investpeople_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)


# In[69]:


plt.figure(figsize=[25, 12])
sns.countplot(x = 'AGE', hue = 'default.invesment.next.month', data = investpeople_df)


# In[70]:


plt.figure(figsize=[20,20])
plt.subplot(311)
sns.countplot(x = 'EDUCATION', hue = 'default.invesment.next.month', data = investpeople_df)
plt.subplot(312)
sns.countplot(x = 'SEX', hue = 'default.invesment.next.month', data = investpeople_df)
plt.subplot(313)
sns.countplot(x = 'MARRIAGE', hue = 'default.invesment.next.month', data = investpeople_df)


# In[71]:


# KDE (Kernel Density Estimate) se utiliza para visualizar la densidad de probabilidad de una variable continua.
# KDE describe la densidad de probabilidad a diferentes valores en una variable continua.

plt.figure(figsize=(12,7))

sns.distplot(cc_nodefault_df['SALARY_AMT'], bins = 250, color = 'r')
sns.distplot(cc_default_df['SALARY_AMT'], bins = 250, color = 'b')

plt.xlabel('Cantidad que debe en cuenta a septiembre de 2005 (dólar NT)')
#plt.xlim(0, 200000)


# In[80]:


# KDE (Kernel Density Estimate) se utiliza para visualizar la densidad de probabilidad de una variable continua.
# KDE describe la densidad de probabilidad a diferentes valores en una variable continua.

plt.figure(figsize=(12,7))

sns.kdeplot(cc_nodefault_df['NO_INVEST_AMT1'], label = 'Clientes sin inversión (cantidad invertida)', shade = True, color = 'r')
sns.kdeplot(cc_default_df['NO_INVEST_AMT1'], label = 'Clientes con inversión (cantidad sin invertir)', shade = True, color = 'b')

plt.xlabel('Cantidad que no invierte en cuenta a septiembre de 2018 (dólar)')
#plt.xlim(0, 200000)


# In[78]:


# KDE (Kernel Density Estimate) is used for visualizing the Probability Density of a continuous variable. 
# KDE describes the probability density at different values in a continuous variable. 

plt.figure(figsize=(12,7))

sns.kdeplot(cc_nodefault_df['INVEST_AMT1'], label = 'Clientes sin inversión (cantidad invertida)', shade = True, color = 'r')
sns.kdeplot(cc_default_df['INVEST_AMT1'], label = 'Clientes con inversión (cantidad sin invertir)', shade = True, color = 'b')

plt.xlabel('PAY_AMT1: Cantidad del pago anterior en septiembre de 2005 (dólar NT)')
plt.xlim(0, 200000)


# In[82]:


# Veamos si el género del individuo tiene efecto en el límite del balance

plt.figure(figsize=[10,20])
plt.subplot(211)
sns.boxplot(x = 'SEX', y = 'SALARY_AMT', data = investpeople_df, showfliers = False)
plt.subplot(212)
sns.boxplot(x = 'SEX', y = 'SALARY_AMT', data = investpeople_df)


# In[83]:


plt.figure(figsize=[10,20])
plt.subplot(211)
sns.boxplot(x = 'MARRIAGE', y = 'SALARY_AMT', data = investpeople_df, showfliers = False)
plt.subplot(212)
sns.boxplot(x = 'MARRIAGE', y = 'SALARY_AMT', data = investpeople_df)


# # # 4: CREACIÓN DEL CONJUNTO DE DATOS DE PRUEBA Y ENTRENAMIENTO Y REALIZAR UNA LIMPIEZA DE DATOS

# In[ ]:


investpeople_df


# In[85]:


X_cat = investpeople_df[['SEX', 'EDUCATION', 'MARRIAGE']]
X_cat


# In[86]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


# In[87]:


X_cat.shape


# In[88]:


X_cat = pd.DataFrame(X_cat)


# In[89]:


X_cat


# In[101]:


# Notemos que hemos eliminado la variable objetivo 'default.invesment.next.month'
X_numerical = investpeople_df[['SALARY_AMT', 'AGE', 'INVEST_0', 'INVEST_2', 'INVEST_3', 'INVEST_4', 'INVEST_5', 
                'NO_INVEST_AMT1','NO_INVEST_AMT2', 'NO_INVEST_AMT3', 'NO_INVEST_AMT4', 'NO_INVEST_AMT5', 'NO_INVEST_AMT6',
                'INVEST_AMT1', 'INVEST_AMT2', 'INVEST_AMT3', 'INVEST_AMT4', 'INVEST_AMT5', 'INVEST_AMT6']]
X_numerical


# In[102]:


X_all = pd.concat([X_cat, X_numerical], axis = 1)
X_all


# In[97]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)


# In[103]:


# 0 SI EL CLIENTE NO INVIERTE EL MES SIGUIENTE 
#1 SI EL CLLIENTE INVIERTE AL MES SIGUIENTE
y = investpeople_df['default.invesment.next.month']
y


# # # 5: COMPRENDER LA TEORÍA Y LA INTUICIÓN DETRÁS DEL ALGORITMO XGBOOST

# # Documentación del algotimo XGBOOST
# 

# Ventajas y desventajas del algoritmo

# # Boosting

# Ejemplo:

# # Esemble Learning

# 

# # Árboles de decisión

# 

# 

# 

# 

# # # 6: COMPRENDER LOS PASOS CLAVE DEL ALGORITMO XGBOOST

# Pasos del algotimo

# 

# # # 7: ENTRENAR Y EVALUAR UN CLASIFICADOR XGBOOST

# In[ ]:


# 75% para entrenar el modelo y el 25% para testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[116]:


X_train.shape


# In[117]:


X_test.shape


# In[118]:


get_ipython().system('pip install xgboost')


# In[119]:


# Entrenar un modelo de regresión con XGBoost 

import xgboost as xgb


model = xgb.XGBClassifier(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 100)

model.fit(X_train, y_train)


# In[120]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)


# In[121]:


y_pred


# In[122]:


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[123]:


# Eficacia en el conjunto de test
# 56000 observaciones correctamente predichas de personas que invierten
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)


# In[127]:


print(classification_report(y_test, y_pred))


# # # 8: OPTIMIZAR LOS HIPERPARÁMETROS de XGBOOST REALIZANDO UN GRID SEARCH

# In[128]:


param_grid = {
        'gamma': [0.5, 1, 5],   # parámetro de regularización
        'subsample': [0.6, 0.8, 1.0], # % de filas que usamos para construir cada árbol
        'colsample_bytree': [0.6, 0.8, 1.0], # % de columnas usadas por cada árbol
        'max_depth': [3, 4, 5] # profundidad de cada árbol
        }


# In[130]:


from xgboost import XGBClassifier

xgb_model = XGBClassifier(learning_rate=0.01, n_estimators=100, objective='binary:logistic')
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(xgb_model, param_grid, refit = True, verbose = 4)
grid.fit(X_train, y_train)


# In[135]:


y_predict_optim = grid.predict(X_test)


# In[136]:


y_predict_optim


# In[137]:


# Eficacia en el conjunto de test
cm = confusion_matrix(y_predict_optim, y_test)
sns.heatmap(cm, annot=True)


# In[134]:


print(classification_report(y_test, y_predict_optim))


# # 9: ALGORITMO XG-BOOST EN AWS SAGEMAKER

# 

# 

# 

# 

# In[139]:


X_train.shape


# # TAREA # 10: ENTRENAR XG-BOOST CON SAGEMAKER

# In[143]:


# Convertir la matriz en un data frame de manera que la variable de destino se establezca como la primera columna y seguida de las columnas de características
# Esto se debe a que el algoritmo incorporado de Sagemaker espera los datos en este formato.

train_data = pd.DataFrame({'Target': y_train})
for i in range(X_train.shape[1]):
    train_data[i] = X_train[:,i]


# In[145]:


train_data.head()


# In[149]:


val_data = pd.DataFrame({'Target':y_test})
for i in range(X_test.shape[1]):
    val_data[i] = X_test[:,i]


# In[150]:


val_data.head()


# In[165]:


val_data.shape


# In[166]:


# Guardamos train_data y validation_data como archivos csv.

train_data.to_csv('train.csv', header = False, index = False)
val_data.to_csv('validation.csv', header = False, index = False)


# In[167]:


# Boto3 es el kit de desarrollo de software (SDK) de Amazon Web Services (AWS) para Python
# Boto3 permite al desarrollador de Python escribir software que hace uso de servicios como Amazon S3 y Amazon EC2

import sagemaker
import boto3

# Crear una sesión de sagemaker
sagemaker_session = sagemaker.Session()

# S3 Bucket y prefijo que queremos usar
# default_bucket: crea un bucket de Amazon S3 que se utilizará en esta sesión
bucket = 'sagemaker-practical-invest'
prefix = 'XGBoost-Regressor'
key = 'XGBoost-Regressor'
# Los roles dan acceso de aprendizaje y al alojamiento a nuestros datos
# Esto se especifica al abrir la instancia de sagemakers en "Crear un rol de IAM"
role = sagemaker.get_execution_role()


# In[168]:


print(role)


# In[169]:


# leer los datos del archivo csv y luego cargar los datos en el depósito s3
import os
with open('train.csv','rb') as f:
    # El siguiente código carga los datos en el bucket de S3 para acceder más tarde al entrenamiento
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(f)

# Imprimamos la ubicación de los datos de entrenamiento en s3
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('ubicación de datos de entrenamiento cargados: {}'.format(s3_train_data))


# In[170]:


# leer los datos del archivo csv y luego cargar los datos en el bucket s3

with open('validation.csv','rb') as f:
    # El siguiente código carga los datos en el bucket de S3 para acceder más tarde al entrenamiento
    boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', key)).upload_fileobj(f)

# Imprimamos la ubicación de los datos de validación en s3
s3_validation_data = 's3://{}/{}/validation/{}'.format(bucket, prefix, key)
print('uploaded validation data location: {}'.format(s3_validation_data))


# In[171]:


# crea un placeholder de salida en el bucket S3 para almacenar la salida

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('los artefactos de entrenamiento se cargarán en: {}'.format(output_location))


# In[172]:


# Este código se usa para obtener el contenedor de entrenamiento de los algoritmos integrados de sagemaker
# todo lo que tenemos que hacer es especificar el nombre del algoritmo que queremos usar

# Obtengamos una referencia a la imagen del contenedor XGBoost
# Hay que tener en cuenta que todos los modelos de regresión se denominan estimadores
# No tenemos que especificar (hardcodear) la región, get_image_uri obtendrá el nombre de la región actual usando boto3.

from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'xgboost','0.90-2') # Latest version of XGboost


# In[182]:


# Especificar el tipo de instancia que nos gustaría usar para el entrenamiento.
# ruta de salida y sesión de sagemaker en el Estimador.
# También podemos especificar cuántas instancias nos gustaría utilizar para el entrenamiento

# Recordemos que XGBoost funciona combinando un conjunto de modelos débiles para generar resultados precisos / sólidos.
# Los modelos débiles son aleatorios para evitar el sobreajuste

# num_round: el número de rondas para ejecutar el entrenamiento.

# Alfa: Término de regularización L1 sobre pesos. Incrementar este valor hace que los modelos sean más conservadores.

# colsample_by_tree: fracción de características que se usarán para entrenar cada árbol.

# eta: Reducción del tamaño del paso que se utiliza en las actualizaciones para evitar el sobreajuste.
# Después de cada paso de boosting, el parámetro eta reduce los pesos de las funciones para hacer que el proceso de impulso sea más conservador.


Xgboost_regressor1 = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.m4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)

# Podemos ajustar los hiperparámetros para mejorar el rendimiento del modelo

Xgboost_regressor1.set_hyperparameters(max_depth = 10,
                           objective = 'multi:softmax',
                           num_class = 2,
                           #colsample_bytree = 0.3,
                           #alpha = 10,
                           eta = 0.5,
                           num_round = 150
                           )


# In[183]:


# Creamos los canales "train", "validation" para entregar los datos al modelo
# Fuente: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html

train_input = sagemaker.session.s3_input(s3_data = s3_train_data, content_type='csv',s3_data_type = 'S3Prefix')
valid_input = sagemaker.session.s3_input(s3_data = s3_validation_data, content_type='csv',s3_data_type = 'S3Prefix')


data_channels = {'train': train_input,'validation': valid_input}


Xgboost_regressor1.fit(data_channels)


# In[194]:


X_test.shape


# 

# In[ ]:





# In[ ]:





# In[ ]:




