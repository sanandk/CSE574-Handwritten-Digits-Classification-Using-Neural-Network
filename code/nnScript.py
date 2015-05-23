import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    s = 1.0 / (1.0 + np.exp(-1.0 * z));
    return  s
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('D:/Projects/ML/basecode/basecode/mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    ftrainset=np.empty((0,784));
    testset=np.empty((0,784))
    testlabel=np.empty((0,1))
    trainset=np.empty((0,784))
    ftrainset=np.empty((0,784))
    trainlabel=np.empty((0,1))
    valset=np.empty((0,784))
    vallabel=np.empty((0,1))
    
    for i in range(10):
        m = mat.get('train'+str(i));
        m2 = mat.get('test'+str(i));
     
        a=range(m.shape[0]);
        cnt=m.shape[0]
        testcnt=m2.shape[0]
        traincnt=int(cnt*5/6)
        valcnt=cnt-traincnt

        aperm=np.random.permutation(a)
        A1=m[aperm[0:traincnt],:]
        A2=m[aperm[traincnt:],:]
  
        ftrainset=np.vstack([ftrainset,m]);
        trainset=np.vstack([trainset,A1]);
        valset=np.vstack([valset,A2]);
        testset=np.vstack([testset,m2]);
        
        for j in range(traincnt):
            trainlabel=np.append(trainlabel,i);
        for j in range(valcnt):
            vallabel=np.append(vallabel,i);   
        for j in range(testcnt):
            testlabel=np.append(testlabel,i);   
  
    # Feature Selection with variance threshhold as p=0.75
  
    selected_features=np.invert(np.all(ftrainset - ftrainset[0,:] < 3 , axis=0))
    trainset=trainset[:,selected_features];
    valset=valset[:,selected_features];
    testset=testset[:,selected_features];
    
    trainset/=255;
    valset/=255;
    testset/=255;
     
    #50000 training, 10000 validation, 10000 testing
    train_data = trainset
    train_label = trainlabel
    validation_data = valset
    validation_label = vallabel
    test_data = testset
    test_label = testlabel
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
   
    label1=np.zeros((training_label.shape[0],10));    

    for i in range (training_label.shape[0]):    
        label1[i][training_label[i]]=1;

    training_data = np.append(training_data,np.zeros([len(training_data),1]),1)
    n=training_data.shape[0]
    
    egrad1=0
    egrad2=0

    z,o=feedfor(training_data,w1,w2);

    y=label1

    delta=o-y;

        #scalar
    scalar=(y * np.log(o)) + ( (1-y) * np.log(1-o))

    D=np.dot(delta,w2)
    # Dimension of D is 50k x 51
    

    grad2=np.dot(delta.T,z)
    z=z*(1-z)     
    D=D*z
    grad1=np.dot(training_data.T,D)

    grad1 = np.delete(grad1, (n_hidden), axis=1)
    e17=grad1.T+(lambdaval*w1);
    e16=grad2+(lambdaval*w2);

       
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    
    egrad1=e17/n
    egrad2=e16/n

    obj_grad = np.concatenate((egrad1.flatten(), egrad2.flatten()),0)
    obj_val= (-np.sum(scalar)/n) + ( (lambdaval/(2 * n)) * ( np.sum(np.square(w1)) + np.sum(np.square(w2)) ) )
    #obj_val=np.sum(obj_val);
    
    return (obj_val,obj_grad)


def feedfor (data,w1,w2):

        a=np.dot(data,w1.T)
          
        z=sigmoid(a);
        z = np.append(z,np.zeros([len(z),1]),1)
        b=np.dot(z,w2.T)        
        o=sigmoid(b);    
        index=np.argmax(o, axis=1);
        label=np.zeros((o.shape[0],10));
        for i in range(label.shape[0]):
            label[i][index[i]]=1;

        return (z,o)

def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    data = np.append(data,np.zeros([len(data),1]),1)
    n=data.shape[0]    
    z,o=feedfor(data,w1,w2);
    ans=np.empty((0,1))
    for i in range(n):
        index=np.argmax(o[i]);
        ans=np.append(ans,index);
    return ans
    

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


pickle.dump( [n_hidden, w1, w2, lambdaval], open( "params.pickle", "wb" ) )

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
