from keras.layers import Activation, Dense,BatchNormalization,Dropout
import keras
from keras import regularizers
my_init=keras.initializers.glorot_uniform(seed=2020)

class ae_model(object):
    
    def __init__(self,N,H_list,drop_rate=0.1,reg=0.01,active=0):
        self.N=N
        self.H_list=H_list
        self.drop_rate=drop_rate
        self.reg=reg
        self.active=active
    ##编码部分     
    def encode(self,x):
        for i,H in enumerate(self.H_list):
            x=Dense(H,activation='relu',name=f'eh{i}',kernel_regularizer=regularizers.l2(self.reg),activity_regularizer=regularizers.l1(self.active))(x)
            #x=BatchNormalization()(x)
            x=Dropout(self.drop_rate)(x)
        return x
    
    ##自编码输出  
    def auto_encode(self,x):
        decoded=self.encode(x)
        for i in range(len(self.H_list)-1,-1,-1):
            H=self.H_list[i]
            decoded=Dense(H,activation='relu')(decoded)
        decoded_out= Dense(self.N,activation='relu')(decoded)
        return decoded_out

    

    
    
