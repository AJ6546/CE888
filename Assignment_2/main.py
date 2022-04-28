'''Importing Libraries required'''
import pandas as pd
import numpy as np
import time
import re
import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#%%
'''Importing Dataset - Creating Dataframe'''
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'train_spider.json')
df = pd.read_json(file_path, orient='records')
cols = ['question','query']
data=df[cols]
data=data.drop_duplicates()
#%%
#Changing Everything to Uppercase
data['query']=data['query'].str.upper()
data['question']=data['question'].str.upper()

data=data.drop_duplicates()
#%%
def DataSetWith1JoinAndWhere():
    df1= pd.DataFrame(columns=['query', 'question'])
    df1=pd.concat([df1,data.loc[data['query'].str.contains('WHERE', case=False)]],ignore_index=True, sort=False)
    df2= pd.DataFrame(columns=['query', 'question'])
    for x in range(len(df1)):
        if(df1.loc[x]['query'].count('JOIN')==1):
            df2=df2.append(df1.loc[x])
    return df2.drop_duplicates().reset_index()
df_final=DataSetWith1JoinAndWhere()[cols].sample(n=1000,random_state=1).reset_index()
del df_final['index']
#%%
# Creating Train Validation Test Split
from fast_ml.model_development import train_valid_test_split

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(df_final, target = 'query', random_state=1,
                                                                            train_size=0.7, valid_size=0.15, test_size=0.15)
#%%
#Cleaning Questions

def clean_text(text):
    text = text.upper()
    text = re.sub(r"WHATS'S", "WHAT IS", text)
    text = re.sub(r"WHERE'S", "WHERE IS", text)
    text = re.sub(r"\'LL", " WILL", text)
    text = re.sub(r"\'VE", " HAVE", text)
    text = re.sub(r"\'RE", " ARE", text)
    text = re.sub(r"\'D", " WOULD", text)
    text = re.sub(r"WON'T", "WILL NOT", text)
    text = re.sub(r"CAN'T", "CANNOT", text)
    text = re.sub(r"[\"#/@;:{}~|.?,]", "", text)
    for i in range(1,10):   
        text = re.sub(r"\ T"+str(i), " ", text)
    return text

clean_questions=[]
for question in X_train['question']:
    clean_questions.append(clean_text(question))

clean_queries=[]
for query in y_train:
    clean_queries.append(query)#clean_text(query)

#%%
#Tokenizing
tokenizer_questions = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(clean_questions,
                                                                                target_vocab_size=2**13)
tokenizer_queries = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(clean_queries,
                                                                                target_vocab_size=2**13)
        
VOCAB_SIZE_QUESTIONS = tokenizer_questions.vocab_size+2
VOCAB_SIZE_QUERIES = tokenizer_queries.vocab_size+2

inputs =[[VOCAB_SIZE_QUESTIONS-2]+tokenizer_questions.encode(sentence)+[VOCAB_SIZE_QUESTIONS-1]
         for sentence in clean_questions]

outputs =[[VOCAB_SIZE_QUERIES-2]+tokenizer_queries.encode(sentence)+[VOCAB_SIZE_QUERIES-1]
         for sentence in clean_queries]

MAX_LENGTH=40
idx_to_remove =[count for count, sent in enumerate(inputs) if len(sent)>MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]
    
idx_to_remove =[count for count, sent in enumerate(outputs) if len(sent)>MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]

#Padding Inputs and Outputs
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                                  value=0,padding='post',
                                                                  maxlen=MAX_LENGTH)
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs,
                                                                  value=0,padding='post',
                                                                  maxlen=MAX_LENGTH)

#%%
#Getting TrainDataset
dataset=tf.data.Dataset.from_tensor_slices((inputs,outputs))
#%%
BATCH_SIZE=64
BUFFER_SIZE=20000

dataset=dataset.cache()
dataset=dataset.shuffle(BUFFER_SIZE).batch(BUFFER_SIZE)
dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)
#%%
# Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding,self).__init__()
    def get_angles(self,pos,i, d_model): #pos:(seq_length,1) i: (1,d_model)
        angles = i/np.power(10000.,(2*(i//2))/np.float32(d_model))
        return pos * angles #(seq_length,d_model)
    def call(self,inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        angles=self.get_angles(np.arange(seq_length)[:,np.newaxis],
                               np.arange(d_model)[np.newaxis,:],
                               d_model)
        angles[:,0::2]=np.sin(angles[:,0::2])
        angles[:,1::2]=np.cos(angles[:,1::2])
        pos_encoding=angles[np.newaxis,...]
        return inputs+tf.cast(pos_encoding,tf.float32)

# Scaled Dot Product Attention
def scaled_dotproduct_attention(quries,keys,values,mask):
    product=tf.matmul(quries,keys,transpose_b=True)
    keys_dim=tf.cast(tf.shape(keys)[-1],tf.float32)
    scaled_product=product/tf.math.sqrt(keys_dim)
    if mask is not None:
        scaled_product +=(mask*-1e9)
    attention=tf.matmul(tf.nn.softmax(scaled_product,axis=-1), values)
    return attention

# Multihead Attention
class MultiHeadAttention(layers.Layer):
    def __init__(self,nb_proj):
        super(MultiHeadAttention,self).__init__()
        self.nb_proj=nb_proj
    def build(self,input_shape):
        self.d_model=input_shape[-1]
        assert self.d_model % self.nb_proj == 0
        
        self.d_proj=self.d_model//self.nb_proj
        
        self.query_lin = layers.Dense(units=self.d_model)
        self.key_lin = layers.Dense(units=self.d_model)
        self.value_lin = layers.Dense(units=self.d_model)
        self.final_lin = layers.Dense(units=self.d_model)
        
    def split_proj(self,inputs,batch_size): # inputs: (batch_size, seq_length,d_model)
        shape = (batch_size,
                 -1,
                 self.nb_proj,
                 self.d_proj)
        splitted_inputs = tf.reshape(inputs,shape=shape) # (batch_size,seq_length,np_proj, d_proj)
        return tf.transpose(splitted_inputs, perm =[0,2,1,3])# (batch_size, np_proj,seq_length,d_proj)
        
    def call(self, queries,keys,values,mask):
        batch_size=tf.shape(queries)[0]
        
        queries=self.query_lin(queries)
        keys=self.key_lin(keys)
        values=self.value_lin(values)
        
        queries = self.split_proj(queries,batch_size)
        keys = self.split_proj(keys,batch_size)
        values = self.split_proj(values,batch_size)

        attention = scaled_dotproduct_attention(queries,keys,values,mask)
        attention = tf.transpose(attention, perm = [0,2,1,3])
        
        concat_attention= tf.reshape(attention,
                                     shape=(batch_size,-1,self.d_model))
        
        outputs = self.final_lin(concat_attention)
        return outputs

class EncoderLayer(layers.Layer):
    
    def __init__(self,FFN_units, nb_proj, dropout):
        super(EncoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout = dropout
        
    def build(self, input_shape):
        self.d_model=input_shape[-1]
        self.multi_head_attention = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.dense_1 = layers.Dense(units=self.FFN_units,activation='relu')
        self.dense_2 = layers.Dense(units=self.d_model,activation='relu')
        self.dropout_2 = layers.Dropout(rate=self.dropout)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask, training):
        attention = self.multi_head_attention(inputs,inputs,inputs,mask)
        attention =self.dropout_1(attention, training=training)
        attention = self.norm_1(attention+inputs)
        outputs=self.dense_1(attention)
        outputs=self.dense_2(outputs)
        outputs =self.dropout_2(outputs)
        outputs = self.norm_2(outputs+attention)
        
        return outputs
    
class Encoder(layers.Layer):
    def __init__(self,
               nb_layers,
               FFN_units,
              nb_proj,
              dropout,
              vocab_size,
              d_model,
              name="encoder"):
        super(Encoder,self).__init__(name=name)
        self.nb_layers=nb_layers
        self.d_model=d_model
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout)
        self.enc_layers = [EncoderLayer(FFN_units,
                                        nb_proj,
                                        dropout)
                           for _ in range(nb_layers)]
        
    def call(self,inputs,mask,training):
        outputs = self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_layers):
            outputs = self.enc_layers[i](outputs, mask, training)
            
        return outputs

class DecoderLayer(layers.Layer):
    def __init__(self,FFN_units, nb_proj, dropout):
        super(DecoderLayer, self).__init__()
        self.FFN_units = FFN_units
        self.nb_proj = nb_proj
        self.dropout = dropout
        
    def build(self, input_shape):
        self.d_model=input_shape[-1]
        self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
        self.dropout_1 = layers.Dropout(rate=self.dropout)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
        
        self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
        self.dropout_2 = layers.Dropout(rate=self.dropout)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dense_1 = layers.Dense(units=self.FFN_units,activation='relu')
        self.dense_2 = layers.Dense(units=self.d_model)
        self.dropout_3 = layers.Dropout(rate=self.dropout)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        attention = self.multi_head_attention_1(inputs,
                                              inputs,
                                              inputs,
                                              mask_1)
        attention = self.dropout_1(attention, training)
        attention = self.norm_1(attention +  inputs)
        
        attention_2 = self.multi_head_attention_2(attention,
                                              enc_outputs,
                                              enc_outputs,
                                              mask_2)
        attention_2 = self.dropout_2(attention_2,training)
        attention_2 = self.norm_2(attention_2 +  attention)
        
        outputs = self.dense_1(attention_2)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs, training)
        outputs = self.norm_3(outputs +  attention_2)
        
        return outputs

class Decoder(layers.Layer):
    def __init__(self,
               nb_layers,
               FFN_units,
              nb_proj,
              dropout,
              vocab_size,
              d_model,
              name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.d_model=d_model
        self.nb_layers = nb_layers
        
        self.embedding =  layers.Embedding(vocab_size,d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout)
        
        self.dec_layers = [DecoderLayer(FFN_units, nb_proj,dropout)
                           for _ in range(nb_layers)]
    def call(self, inputs, enc_outputs, mask_1, mask_2, training):
        outputs=self.embedding(inputs)
        outputs *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        outputs = self.pos_encoding(outputs)
        outputs = self.dropout(outputs,training)
        
        for i in range(self.nb_layers):
            outputs = self.dec_layers[i](outputs,
                                         enc_outputs,
                                         mask_1,
                                         mask_2,
                                         training)
        return outputs
    
# Transformer
class Transformer(tf.keras.Model):
    def __init__(self,vocab_size_enc,
              vocab_size_dec,
              d_model,
              nb_layers,
              FFN_units,
              nb_proj,
              dropout,
              name="transformer"):
        super(Transformer, self).__init__(name=name)
        
        self.encoder = Encoder(nb_layers,
                              FFN_units,
                              nb_proj,
                              dropout,
                              vocab_size_enc,
                              d_model)
        self.decoder = Decoder(nb_layers,
                              FFN_units,
                              nb_proj,
                              dropout,
                              vocab_size_dec,
                              d_model)
        
        self.last_linear = layers.Dense(units=vocab_size_dec)
        
    def create_padding_mask(self, seq): # seq: (batch_size,seq_length)
        mask = tf.cast(tf.math.equal(seq,0),tf.float32)
        return mask[:,tf.newaxis,tf.newaxis,:]
    
    def create_look_ahead_mask(self, seq):
        seq_len = tf.shape(seq)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len,seq_len)),-1,0)
        return look_ahead_mask
    
    def call(self, enc_inputs,dec_inputs, training):
        enc_mask=self.create_padding_mask(enc_inputs)
        dec_mask_1= tf.maximum(self.create_padding_mask(dec_inputs), 
                               self.create_look_ahead_mask(dec_inputs))
        dec_mask_2 = self.create_padding_mask(enc_inputs)
        
        enc_outputs = self.encoder(enc_inputs,enc_mask, training)
        dec_outputs = self.decoder(dec_inputs,
                                   enc_outputs,
                                   dec_mask_1,
                                   dec_mask_2,
                                   training)
        
        outputs = self.last_linear(dec_outputs)
        return outputs
#%%
# Training
tf.keras.backend.clear_session()

#Hyper parameters
D_MODEL=128 # 512
NB_LAYERS=6 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROUPOUT = 0.01 # 0.1

transformer = Transformer(vocab_size_enc = VOCAB_SIZE_QUESTIONS,
                          vocab_size_dec = VOCAB_SIZE_QUERIES,
                          d_model= D_MODEL,
                          nb_layers=NB_LAYERS,
                          FFN_units=FFN_UNITS,
                          nb_proj=NB_PROJ,
                          dropout=DROUPOUT)
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")

def loss_funstion(target, pred):
    mask=tf.math.logical_not(tf.math.equal(target, 0))
    loss_=loss_object(target,pred)
    
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask
    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule,self).__init__()
        
        self.d_model = tf.cast(d_model,
                               tf.float32)
        self.warmup_steps=warmup_steps
        
    def __call__(self,step):
        arg1=tf.math.rsqrt(step)
        arg2=step*(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)
    
learning_rate=CustomSchedule(D_MODEL)
optimizer=tf.keras.optimizers.Adam(learning_rate,
                                   beta_1=0.9,
                                   beta_2=0.98,
                                   epsilon=1e-9)

checkpoint_path =os.path.join(script_dir, "Checkpoints")

ckpt=tf.train.Checkpoint(transformer=transformer,
                   optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest Checkpoint Restored")
def evaluate(inp_sentence):
    inp_sentence = [VOCAB_SIZE_QUESTIONS-2] + tokenizer_questions.encode(inp_sentence) + [VOCAB_SIZE_QUESTIONS-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
        
    output = tf.expand_dims([VOCAB_SIZE_QUERIES-2], axis=0)
    
    for _ in range(MAX_LENGTH):
        predictions = transformer(enc_input, output, False) # 1, seq_length, VOCAB_SIZE_QUERIES
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == VOCAB_SIZE_QUERIES-1:
            return tf.squeeze(output,axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)

def ChangeUndsc(predicted_sentence):
    output=predicted_sentence.replace('\&undsc','_')
    return output

def translate(sentence):
    output = evaluate(clean_text(sentence)).numpy()
    predicted_sentence = tokenizer_queries.decode(
        [i for i in output if i < VOCAB_SIZE_QUERIES-2])
    output = ChangeUndsc(predicted_sentence)
    warnings.filterwarnings("ignore")
    print("\nInput: {}".format(sentence))
    print("\nPredicted Translation : {}\n\n".format(output))
#%%
# Insert Question here
question='Show the times used by climbers to climb mountains in Country Uganda.'
translate(question)
question= input('Enter Question: ')
translate(question)