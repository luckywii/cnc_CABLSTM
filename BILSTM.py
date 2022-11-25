import sklearn.model_selection
import sklearn.metrics
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.layers import LSTM, Dense, Activation, Bidirectional
from keras_self_attention import SeqSelfAttention
from keras import optimizers
#from attention_decoder import AttentionDecoder
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category = FutureWarning)

main_df = pd.read_csv('C:/Users/lee/Desktop/인턴 데이터셋 자료/CNC Milling Dataset/train.csv')
main_df = main_df.fillna('no')
main_df.head()

#print(main_df)

experiment_01 = pd.read_csv('C:/Users/lee/Desktop/인턴 데이터셋 자료/CNC Milling Dataset/experiment_01.csv')
experiment_01.head()

# print(experiment_01)

files = list()

for i in range(1, 19):
    exp_number = '0' + str(i) if i < 10 else str(i)
    file = pd.read_csv('C:/Users/lee/Desktop/인턴 데이터셋 자료/CNC Milling Dataset/experiment_{}.csv'.format(exp_number))
    ## experiment 1,2,3,4,~~ 18 전부 읽어오기
    row = main_df[main_df['No'] == i]

    ## add experiment settings to feature
    file['feedrate'] = row.iloc[0]['feedrate']
    file['clamp_pressure'] = row.iloc[0]['clamp_pressure']

    ## havingf label as 'tool_condition'

    file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    files.append(file)

    #####experiment 폴더에 feedrate. clamp_pressure, worn을 feature로 추가 worn --> label 1 , unworn --> label 0


df = pd.concat(files, ignore_index = True)
### 파일에 있는 모든 experiment파일을 하나로 concat시킴  ### (25286, 51)
#df = shuffle(df)


process = {'Layer 1 Up': 1, 'Repositioning': 2, 'Layer 2 Up': 3, 'Layer 2 up': 4,
       'Layer 1 Down': 5, 'End': 6, 'Layer 2 Down': 7, 'Layer 3 Down': 8, 'Prep': 9,
       'end': 10, 'Starting': 11}

data = [df]

for dataset in data:
    dataset['Machining_Process'] = dataset['Machining_Process'].map(process)

# df = df.drop(['Z1_CurrentFeedback', 'Z1_DCBusVoltage', 'Z1_OutputCurrent', 'Z1_OutputVoltage',
#               'S1_SystemInertia'], axis = 1)

corm = df.corr()
corm

X = df.drop(['label', 'Machining_Process'], axis = 1)
Y = df['label']

X = X.to_numpy() #### (25286, 44)
Y = Y.to_numpy() #### (25286,)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 2020) ### random_state 데이터 분할시 셔플이 이루어지는데 이를 위한 시드값

y_train = to_categorical(y_train) #### to_categorical 원핫인코딩으로 변환
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 49, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 49, 1).astype('float32')

# seq_length = 54

model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(49,  1))) ### Conv1D(number of filters, kernel_size,
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(SeqSelfAttention(attention_activation = 'softmax'))
model.add(Bidirectional(LSTM(256, return_sequences = False))) ## LSTM(units, ....)
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
sgd = optimizers.Adam(lr = 0.001, decay = 0.0, beta_1=0.9, beta_2=0.999, epsilon=0.01, amsgrad=False)
model.compile(loss = 'binary_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

model.summary()


early_stopping = EarlyStopping(monitor = 'val_loss', patience = 4)
hist = model.fit(x_train, y_train, batch_size = 256, epochs = 1000, validation_split = 0.2)

plt.figure(figsize = (12, 4))

plt.subplot(1, 3, 1)
plt.plot(hist.history['loss'], 'b-', label = 'loss')
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist.history['accuracy'], 'g-', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'r-', label='validation accuracy')
plt.ylim(0, 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 3, 3)
plt.plot(hist.history['accuracy'], 'g-', label='accuracy')
plt.plot(hist.history['val_accuracy'], 'r-', label='validation accuracy')
plt.ylim(0.8, 1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()