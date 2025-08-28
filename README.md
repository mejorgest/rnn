# rnn
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report  

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global = 1 - precision_global
  precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis = 1)).T
  if nombres != None:
    precision_categoria.columns = nombres
  
  return {
    "Matriz de Confusión" : MC,
    "Precisión Global" : precision_global,
    "Error Global" : error_global,
    "Precisión por categoría" : precision_categoria}

alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

from sklearn.preprocessing import LabelEncoder

alfabeto = list(alfabeto)
encoder = LabelEncoder()
datos = encoder.fit_transform(alfabeto)
datos


# creamos nuestros pares de entrada y salida para entrenar nuestra red neuronal
X = np.array([])
y = np.array([])

for i in range(len(datos) - 1):
  entrada = datos[i]
  salida  = datos[i + 1]
  X = np.append(X, entrada)
  y = np.append(y, salida)
  print(encoder.inverse_transform([entrada]), '->', encoder.inverse_transform([salida]))

X = np.reshape(X, (len(X), 1, 1))
print("Shape: ", X.shape)
print(X)

# creamos nuestros pares de entrada y salida para entrenar nuestra red neuronal
X = X / float(len(alfabeto))
X

y = to_categorical(y)
print(y[0:5])

# Vamos a crear una red LSTM con 32 unidades y una salida
# función de activación de softmax para hacer predicciones
# la función de pérdida de registro (llamada cruzamiento categórico o categorical_crossentropy en Keras) 
# función de optimización de ADAM
modelo = Sequential()
modelo.add(LSTM(64, input_shape = (X.shape[1], X.shape[2])))
modelo.add(Dense(y.shape[1], activation = 'softmax'))

modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

modelo.summary()

# Se ajusta con 500 epochs con un tamaño de lote de 1.
modelo.fit(X, y, epochs=100, batch_size=1, verbose=0)

pred = modelo.predict(X, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred
y_test = np.argmax(y, axis = 1)
y_test = encoder.inverse_transform(y_test)

MC = confusion_matrix(y_test, pred, labels = encoder.classes_)
indices = indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s" % (k, str(indices[k])))

modelo = Sequential()
modelo.add(LSTM(64, input_shape = (X.shape[1], X.shape[2])))
modelo.add(Dense(y.shape[1], activation = 'softmax'))

modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X, y, epochs=500, batch_size=1, verbose=0)
modelo.save("rnn_alfabeto.h5py")


#cargar modelo

from tensorflow.keras.models import load_model

modelo = load_model('rnn_alfabeto.h5py')
nuevos = "TDOFL"
nuevos = list(nuevos)
nuevos = encoder.transform(nuevos)
nuevos
nuevos = np.reshape(nuevos, (len(nuevos), 1, 1))
nuevos = nuevos / float(len(datos))
nuevos
pred = modelo.predict(nuevos, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred

!pip install librosa


import librosa

acorde_mayor,sample_rate = librosa.load("../../../datos/acorde/mayor/Major_0.wav", sr = 1600)


librosa.get_duration(y = acorde_mayor, sr = 1600)

fig, ax = plt.subplots()
ax.plot(acorde_mayor)
ax.set_xlabel('Tiempo')
ax.set_ylabel('Amplitud')
plt.show()

#librosa nos provee de un conjunto de técnicas las cuales no sirven para obtener características (variables) de un audio, #de manera que podamos reducir la cantidad de variables a analizar eliminando aquellas que no aportan o aportan poca #información, como por ejemplo ruidos de fondo.

#MFC (Mel Frequency cepstrum): Es una representación del espectro de potencia a corto plazo de un sonido, basado en una #transformada de coseno lineal de un espectro de potencia logarítmica en una escala de frecuencia mel no lineal. Es decir, #es una transformación matemática que ayuda a procesar el sonido.



mfc = librosa.feature.melspectrogram(y = acorde_mayor, sr= 1600, n_mels = 8)
mfc

mfc_mean = np.mean(mfc.T, axis=0)
mfc_mean

mfcc = librosa.feature.mfcc(y = acorde_mayor, sr= 1600, n_mfcc = 8)
mfcc

mfcc_mean = np.mean(mfcc.T, axis=0)
mfcc_mean

chroma = librosa.feature.chroma_stft(y = acorde_mayor, sr= 1600, n_chroma = 8)
chroma

chroma_mean = np.mean(chroma.T, axis=0)
chroma_mean



#Para este ejemplo se va a utilizar los datos de acordes los cuales tiene sonidos de acordes de guitarra, donde un acorde #es un conjunto de notas que se dividen en mayores y menores, por tanto el objetivo es identificar si el acorde es mayor o #menor.

labels = []
audios = []
ruta = "../../../datos/acorde/"

for carpeta in next(os.walk(ruta))[1]:
  for nombrearchivo in next(os.walk(ruta + '/' + carpeta))[2]:
    if re.search("\\.(mp3|wav|m4a|wma|aiff)$", nombrearchivo):
      try:
        audio, sample_rate = librosa.load(ruta + '/' + carpeta + '/' + nombrearchivo, sr = 1600)
        audio = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 128)
        audio = np.mean(audio.T, axis=0)
        audios.append(audio)
        labels.append(carpeta)
      except:
        print("No se pudo cargar el audio: " + nombrearchivo + " en la carpeta: " + carpeta)
X = np.array(audios, dtype = np.float32)
y = np.array(labels)
print(
  'Total de individuos: ', len(X),
  '\nNúmero total de salidas: ', len(np.unique(y)), 
  '\nClases de salida: ', np.unique(y))

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)


# Dividir en train y test
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.05)

# Normalizar
scaler = StandardScaler()
sc_training = scaler.fit_transform(train_X)
sc_testing  = scaler.fit_transform(test_X)

modelo = Sequential()

modelo.add(LSTM(units = 128, input_shape = (128, 1)))
modelo.add(Dense(256, activation = "relu"))
modelo.add(Dense(2, activation = "softmax"))

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modelo.summary()
modelo.fit(sc_training, train_Y, epochs=50, batch_size=32, verbose=0)


pred = modelo.predict(sc_testing, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred

test_Y = np.argmax(test_Y, axis = 1)
test_Y = encoder.inverse_transform(test_Y)

MC = confusion_matrix(test_Y, pred, labels = encoder.classes_)
indices = indices_general(MC, list(encoder.classes_))
for k in indices:
  print("\n%s:\n%s" % (k, str(indices[k])))

modelo = Sequential()

modelo.add(LSTM(units = 128, input_shape = (128, 1)))
modelo.add(Dense(256, activation = "relu"))
modelo.add(Dense(2, activation = "softmax"))
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X, y, epochs=50, batch_size=32, verbose=0)
modelo.save("lstm_acordes.h5py")

from tensorflow.keras.models import load_model

modelo = load_model('lstm_acordes.h5py')

la_mayor,sample_rate = librosa.load("../../../datos/prueba/la_mayor.wav", duration = 4, sr = 1600)
la_mayor = librosa.feature.mfcc(y = la_mayor, sr= 1600, n_mfcc = 128)
la_mayor = np.mean(la_mayor.T, axis = 0)

la_menor,sample_rate = librosa.load("../../../datos/prueba/la_menor.wav", duration = 4, sr = 1600)
la_menor = librosa.feature.mfcc(y = la_menor, sr= 1600, n_mfcc = 128)
la_menor = np.mean(la_menor.T, axis = 0)

re_menor,sample_rate = librosa.load("../../../datos/prueba/re_menor.wav", duration = 4, sr = 1600)
re_menor = librosa.feature.mfcc(y = re_menor, sr= 1600, n_mfcc = 128)
re_menor = np.mean(re_menor.T, axis = 0)

re_mayor,sample_rate = librosa.load("../../../datos/prueba/re_mayor.wav", duration = 4, sr = 1600)
re_mayor = librosa.feature.mfcc(y = re_mayor, sr= 1600, n_mfcc = 128)
re_mayor = np.mean(re_mayor.T, axis = 0)

audios = np.array([la_mayor, la_menor, re_menor, re_mayor])

fig, ax = plt.subplots(4, 1, figsize=(6, 6), dpi=100)
for i, axi in enumerate(ax.flat):
    axi.set(xticks=[], yticks=[])
    axi.plot(audios[i])

fig.supxlabel('Tiempo')
fig.supylabel('Amplitud')
plt.show().

audios = scaler.fit_transform(audios)

pred = modelo.predict(audios, verbose = 0)
pred = np.argmax(pred, axis = 1)
pred = encoder.inverse_transform(pred)
pred





datos = pd.read_csv("../../../datos/lyrics.csv", sep = ";")
datos = datos.iloc[0:300, :]
datos

lineas = []

for cancion in datos.text.to_list():
  for linea in cancion.split("\n"):
    linea = linea.lower()
    linea = re.sub(r'[^\w\s]', '', linea)
    linea = re.sub(r'^\s*', '', linea)
    linea = re.sub(r'\s*$', '', linea)
    if linea != "":
      lineas.append(linea)

lineas[0:6]


from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lineas)

dict(list(tokenizer.word_index.items())[0:10])

secuencias = []

for linea in lineas:
  lista_tokens = tokenizer.texts_to_sequences([linea])[0]
  for i in range(2, (len(lista_tokens) + 1)):
    secuencias.append(lista_tokens[:i])

secuencias[:6]

from keras_preprocessing.sequence import pad_sequences

max_palabras   = max([len(x) for x in secuencias])
total_palabras = (len(tokenizer.word_index) + 1)

secuencias = np.array(pad_sequences(secuencias, maxlen = max_palabras, padding = 'pre'))
secuencias[0:9]

X = secuencias[:, :-1]
y = secuencias[:,-1]
y = to_categorical(y, num_classes = total_palabras)

from keras.layers import Embedding, LSTM, Dense, Dropout

modelo = Sequential()

modelo.add(Embedding(total_palabras, 10, input_length = max_palabras - 1))
modelo.add(LSTM(128))
modelo.add(Dropout(0.2))
modelo.add(Dense(total_palabras, activation='softmax'))

modelo.compile(loss='categorical_crossentropy', optimizer='adam')
modelo.summary()
from tensorflow.keras.models import load_model
modelo = load_model('generar_texto.h5py')
nuevo = tokenizer.texts_to_sequences(["fire"])[0]
nuevo = pad_sequences([nuevo], maxlen = max_palabras - 1, padding = 'pre')
predicted = modelo.predict(nuevo, verbose=0)
predicted = np.argmax(predicted, axis=1)
predicted = [k for k, v in tokenizer.word_index.items() if v == predicted[0]][0]
predicted

n = 20
palabras = "fire"

for i in range(n):
  nuevo = tokenizer.texts_to_sequences([palabras])[0]
  nuevo = pad_sequences([nuevo], maxlen = max_palabras - 1, padding = 'pre')
  predicted = modelo.predict(nuevo, verbose=0)
  predicted = np.argmax(predicted, axis=1)
  palabra = [k for k, v in tokenizer.word_index.items() if v == predicted[0]][0]
  palabras = palabras + ' ' + str(palabra)

palabras










