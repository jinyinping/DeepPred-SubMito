from process2 import names,seqs,labels
import numpy as np
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,KFold,train_test_split,cross_val_score,cross_validate, ShuffleSplit
from sklearn.metrics import plot_confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score,f1_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef,auc
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.recurrent import GRU,LSTM
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation,Flatten,Dropout,Dense,Reshape
from keras.optimizers import Adam
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

def process_sample(seq_all,label_all):
    print('Original dataset shape %s' % Counter(label_all))
    #ros = RandomOverSampler(random_state=62,ratio={1:174,2:282,3:174,4:200})
    ros = RandomOverSampler(random_state=62,ratio={1:282,2:282,3:174,4:282})
    #ros = RandomOverSampler(random_state=62)
    X_res, y_res = ros.fit_resample(np.array(seq_all).reshape(-1, 1), label_all)
    print('Resampled dataset shape %s' % Counter(y_res))
    X_res = [str(x) for x in X_res]
    return X_res,y_res

seqs,labels = process_sample(seqs,labels)
#填充序列
def padding_seq(seq,max_len = 180,repkey='N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len - seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

# 剪切序列
def split_overlap_seq(seq,window_size=180):
    overlap_size = 30
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)//(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end -overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seql = seq
        pad_seq = padding_seq(seql,max_len=window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seql = seq[-window_size:]
            pad_seq = padding_seq(seql,max_len = window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

#seq One-hot
def get_protein_cnn_array(seq,motif_len=20):
    alpha = 'ARNDCQEGHILKMFPSTWYV'
    row = (len(seq) + 2*motif_len -2)
    new_array = np.zeros((row,20))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*20)
    for i in range(row-3,row):
        new_array[i] = np.array([0.25]*20)

    for i,val in enumerate(seq):
        i = i + motif_len -1
        if val not in alpha:
            new_array[i] = np.array([0.25]*20)
            continue
        else:
            index = alpha.index(val)
            new_array[i][index] = 1
    #print(np.array(new_array).shape)
    return new_array


def get_bag_data(data,channel=7,window_size=180):
    bags = []
    bag_seqs = []
    X_array = []
    for seq in data:
        bag_seqs = split_overlap_seq(seq)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_protein_cnn_array(bag_seq)
            #print(tri_fea.T)
            bag_subt.append(tri_fea)
        num_of_ins = len(bag_subt)
        #print(num_of_ins)

        if num_of_ins > channel:
            start = (num_of_ins - channel) // 2
            bag_subt = bag_subt[start: start + channel]
        if num_of_ins < channel:
            read_more = channel - len(bag_subt)
            for ind in range(read_more):
                tri_fea = get_protein_cnn_array('N'*window_size)
                bag_subt.append(tri_fea)
        bags.append(np.array(bag_subt))
    #print(len(bags))
    return bags

X_array = get_bag_data(seqs)

#label One-hot
def get_label(label):
    y_array = np.zeros((len(label),4))
    for i,val in enumerate(label):
        t = int(val) - 1
        y_array[i][t] = 1
    return y_array


# print(y.shape)
X_short = np.array(X_array)
print(X_short.shape)

#label One-hot
def get_label(label):
    y_array = np.zeros((len(label),3))
    for i,val in enumerate(label):
        t = int(val) - 1
        y_array[i][t] = 1
    return y_array

X_short_new = np.array(X_array)
print(X_short_new.shape)

print("=============================")

X_short1 = []
for sample in X_short_new:
  X_short1.append(sample.T)

X_short2 = []
for sample in X_short1:
  sample2 = []
  for one in sample:
    sample2.append(one.T)
  sample2 = np.array(sample2)
  X_short2.append(sample2)
X_short3 = np.array(X_short2)

X_short4 = []
for sample in X_short3:
  X_short4.append(sample.T)

X_short4 = np.array(X_short4)


X_short5 = []
for sample in X_short4:
  sample2 = []
  for one in sample:
    sample2.append(one.T)
  sample2 = np.array(sample2)
  X_short5.append(sample2)
X_short = np.array(X_short5)

print(X_short.shape)
y = get_label(labels)
print(y.shape)


class CNN:
    def build(classes, stride=(1, 1)):
        model = Sequential()
        model.add(Conv2D(
            64,
            kernel_size=(5, 5),
            padding='same',
            input_shape=(238, 20, 7)
        ))
        model.add(Activation('linear'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

        # model.add(Conv2D(
        #     128,
        #     kernel_size=(5, 5),
        #     padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(3, 2), strides=(2, 2), padding='same'))
        # model.add(Dropout(0.25))
        model.add(Flatten())
        # model.add(Dense(512, activation='relu'))
        # model.add(Dropout(0.25))
        #model.add(Dense(64))
        model.add(Activation('linear'))
        # model.add(Dropout(0.25))
        # softmax分类器
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.summary()
        return model


def plotAUC(test, score,i):
    fpr, tpr, thresholds = roc_curve(test.ravel(), score.ravel())
    roc_auc = auc(fpr,tpr)
    la = ['The First', 'The Second', 'The Third', 'The fourth', 'The Fifth']
    if i == 0:
        plt.figure(figsize=(5, 5))
        plt.title('Imbalanced Data ROC Curve', fontdict={'family': 'Times New Roman'})
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.plot(fpr, tpr, label=la[i]+' ROC curve (area = %0.3f)' % roc_auc)
    elif i == 4:
        plt.plot(fpr, tpr, label=la[i]+' ROC curve (area = %0.3f)' % roc_auc)
        plt.legend()
        plt.savefig('figure/SubMitoPred_roc_mix.png',)
        plt.show()
    else:
        plt.plot(fpr, tpr, label=la[i]+' ROC curve (area = %0.3f)' % roc_auc)
        plt.legend()

def plotconfu(test, pred,title):
    cfm = confusion_matrix(test, pred)
    cm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]    # 归一化
    labels_name = ['Inner membrane','Matrix','Outer membrane']
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    #plt.matshow(cfm, cmap=plt.cm.Blues)
    #plt.imshow(cfm)
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.show()



def sensitivity(mat):
    tp = np.asarray(np.diag(mat).flatten(), dtype='float')
    fp = np.asarray(np.sum(mat, axis=0).flatten(), dtype='float') - tp
    fn = np.asarray(np.sum(mat, axis=1).flatten(), dtype='float') - tp
    tn = np.asarray(np.sum(mat) * np.ones(3).flatten(),dtype='float') - tp - fn - fp
    sn = tp / (tp + fn)
    sp = tn / ( tn + fp )
    acc = (tp + tn) / (tp + fn + tn + fp)
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator

    return sn,sp,acc,mcc


skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
train_p = []
test_p = []
for i, (train,test) in enumerate(skf.split(X_short,labels)):
    train_p.append(train)
    test_p.append(test)
print(train_p[i].shape)
print(test_p[i].shape)
y = get_label(labels)


mcc_all = []
acc_all = []
f1_all = []
recall_all = []
pre_all = []
sens_all = []
sp_all = []
accuracy_all = []
rouned_labes_all = []
prediction_all = []

predictied_class_all = np.zeros((y.shape[0],))
predictied_all = np.zeros((y.shape))

model_names = ['sbmit1','sbmit2','sbmit3','sbmit4','sbmit5']
#Xs_tr, Xs_lef, ys_tr, ys_lef = train_test_split(X_short, y, test_size=0.1, random_state=4)
for i in range(10):
    Xs_train, Xs_test, ys_train, ys_test = X_short[train_p[i]],X_short[test_p[i]],y[train_p[i]],y[test_p[i]]

    print("time",i)
    print(Xs_test.shape)
    print(Xs_train.shape)

    print('shuffling the data...')
    index = np.arange(len(ys_train))


    Xs_train = Xs_train[index]
    ys_train = ys_train[index]
    ys_test = ys_test


    NB_EPOCH = 150
    BATCH_SIZE = 128
    # 日志显示
    VERBOSE = 30
    OPTIMIZER = Adam(lr=0.001)
    VALIDATION_SPLIT = 0.1
    NB_CLASSES = 3

    model2 = CNN.build(classes=NB_CLASSES)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=OPTIMIZER,
                   metrics=['accuracy'])
    print('Traing...')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=2)
    callbacks_list = [early_stopping,reduce_lr]
    history = model2.fit(Xs_train,
                         ys_train,
                         batch_size=BATCH_SIZE,
                         epochs=NB_EPOCH,
                         verbose=VERBOSE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=callbacks_list
                         )
    print('Evaluating the model')
    prediction = model2.predict(Xs_test)
    predictied_class = model2.predict_classes(Xs_test)
    rouned_labes = np.argmax(ys_test,axis=1)
    #print('rouned_labes',rouned_labes)

    predictied_all[test_p[i]] = prediction
    predictied_class_all[test_p[i]] = predictied_class

    target_names = ['Inner membrane','Matrix','Outer membrane']
    #print(classification_report(rouned_labes, predictied_class, target_names=target_names))
    cm = confusion_matrix(rouned_labes,prediction.argmax(axis=1))
    #print(cm)
    plotconfu(rouned_labes,prediction.argmax(axis=1),'m954_data')
    #plot_confusion_matrix(model2,Xs_test,ys_test,cmap=plt.cm.Blues)

    if i <= 0:
      rouned_labes_all = rouned_labes
      prediction_all = prediction.argmax(axis=1)
    else:
      rouned_labes_all = np.append(rouned_labes_all,rouned_labes)
      prediction_all = np.append(prediction_all,prediction.argmax(axis=1))

    print("=================")
    print(rouned_labes)
    print(predictied_class)
    print("=================")
    print("ACC: %f "%accuracy_score(rouned_labes,predictied_class))
    acc_all.append(accuracy_score(rouned_labes,predictied_class))
    print("F1: %f "%f1_score(rouned_labes,predictied_class,average='micro'))
    f1_all.append(f1_score(rouned_labes,predictied_class,average='micro'))
    print("Recall: %f "%recall_score(rouned_labes,predictied_class,average='micro'))
    recall_all.append(recall_score(rouned_labes,predictied_class,average='micro'))
    print("Pre: %f "%precision_score(rouned_labes,predictied_class,average='micro'))
    pre_all.append(precision_score(rouned_labes,predictied_class,average='micro'))
    
    sn,sp,acc,mcc = sensitivity(cm)
    mcc_all.append(mcc)
    print("MCC: ", mcc)
    sens_all.append(sn)
    print('Sensitivity: ', sn)
    sp_all.append(sp)
    print('SP: ', sp)
    print('accuracy: ', acc)
    mean(accuracy_all.append(acc))
    #plotAUC(ys_test,predicted_prob,i)

    model2.save('/content/sample_data/data/model/m983a_'+str(i)+'.h5')
