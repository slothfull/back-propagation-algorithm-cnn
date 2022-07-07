% --- LeNet-5 inference & backpropagation matlab ---
% --- 117082910078-donghao ---

% --- structure of LeNet_5 ---
% input layer -> 28*28
% convolution layer1 -> 24*24*20
% rectified function = tanh()
% pooling layer1 -> 12*12*20
% full-connected layer -> 10010
% softmax layer -> (softmax 10 for digital numbers)

% --- dataset info ---
% training dataset = 800x10 samples
% testing dateset = 200x10 samples

clear all;
clc;
% --- data preprocessing ---
disp('preprocessing data......')
total_data = load('/Users/mac/Desktop/2/LeNet-5/mnist_data/mnist.mat');
num_sample_to_get_train = 80;
num_sample_to_get_test = 20;
train_data_0 = double(total_data.train0);
train_data_1 = double(total_data.train1);
train_data_2 = double(total_data.train2);
train_data_3 = double(total_data.train3);
train_data_4 = double(total_data.train4);
train_data_5 = double(total_data.train5);
train_data_6 = double(total_data.train6);
train_data_7 = double(total_data.train7);
train_data_8 = double(total_data.train8);
train_data_9 = double(total_data.train9);
test_data_0 = double(total_data.test0);
test_data_1 = double(total_data.test1);
test_data_2 = double(total_data.test2);
test_data_3 = double(total_data.test3);
test_data_4 = double(total_data.test4);
test_data_5 = double(total_data.test5);
test_data_6 = double(total_data.test6);
test_data_7 = double(total_data.test7);
test_data_8 = double(total_data.test8);
test_data_9 = double(total_data.test9);
train_data_0 = train_data_0(1:num_sample_to_get_train,:);
train_data_1 = train_data_1(1:num_sample_to_get_train,:);
train_data_2 = train_data_2(1:num_sample_to_get_train,:);
train_data_3 = train_data_3(1:num_sample_to_get_train,:);
train_data_4 = train_data_4(1:num_sample_to_get_train,:);
train_data_5 = train_data_5(1:num_sample_to_get_train,:);
train_data_6 = train_data_6(1:num_sample_to_get_train,:);
train_data_7 = train_data_7(1:num_sample_to_get_train,:);
train_data_8 = train_data_8(1:num_sample_to_get_train,:);
train_data_9 = train_data_9(1:num_sample_to_get_train,:);
test_data_0 = test_data_0(1:num_sample_to_get_test,:);
test_data_1 = test_data_1(1:num_sample_to_get_test,:);
test_data_2 = test_data_2(1:num_sample_to_get_test,:);
test_data_3 = test_data_3(1:num_sample_to_get_test,:);
test_data_4 = test_data_4(1:num_sample_to_get_test,:);
test_data_5 = test_data_5(1:num_sample_to_get_test,:);
test_data_6 = test_data_6(1:num_sample_to_get_test,:);
test_data_7 = test_data_7(1:num_sample_to_get_test,:);
test_data_8 = test_data_8(1:num_sample_to_get_test,:);
test_data_9 = test_data_9(1:num_sample_to_get_test,:);


% --- use cat() to concentrate the train&test matrix ---
preprocessed_train_data = cat(1,train_data_0,train_data_1,train_data_2,train_data_3...
,train_data_4,train_data_5,train_data_6,train_data_7,train_data_8...
,train_data_9);
preprocessed_test_data = cat(1,test_data_0,test_data_1,test_data_2,test_data_3...
,test_data_4,test_data_5,test_data_6,test_data_7,test_data_8...
,test_data_9);

% --- initial LeNet structure ---
layer_c1_num = 20;
layer_s1_num = 20;
layer_f1_num = 100;
layer_output_num = 10;

% --- learning rate ---
yita = 0.01;

% --- initial biases ---
bias_c1 = (2*rand(1,20)-ones(1,20))/sqrt(20);
bias_f1 = (2*rand(1,100)-ones(1,100))/sqrt(20);

% --- initial kernels ---
[kernel_c1,kernel_f1] = init_kernel(layer_c1_num,layer_f1_num);

% --- initial kernels for pooling ---
pooling_a = ones(2,2)/4;

% --- initial weights for fc-layers --- 
weight_f1 = (2*rand(20,100)-ones(20,100))/sqrt(20);
weight_output = (2*rand(100,10)-ones(100,10))/sqrt(100);
disp('network successfully initialized......');

% --- LeNet training ---
disp('training networks......');
tic
% --- iter = num of epochs ---
for iter=1:20
% each number of character has 800 samples for training
for n=1:num_sample_to_get_train
    for m=0:9
        % reshape data
        train_data = reshape(preprocessed_train_data(m*num_sample_to_get_train+n,:),[28,28]);
        % data normalize
        % train_data = wipe_off_average(train_data);
        % --- inference ---
        for k=1:layer_c1_num
            state_c1(:,:,k) = convolution(train_data,kernel_c1(:,:,k));
            % rectified layer
            state_c1(:,:,k) = tanh(state_c1(:,:,k)+bias_c1(1,k));
            % pooling layer
            state_s1(:,:,k) = pooling(state_c1(:,:,k),pooling_a);
        end
        % full-connected layer
        [state_f1_pre,state_f1_temp] = convolution_f1(state_s1,kernel_f1,weight_f1);
        % rectifier function tanh()
        for nn=1:layer_f1_num
            state_f1(1,nn) = tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        % softmax layer
        for nn=1:layer_output_num
            output(1,nn) = exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
        % compute the train-loss
        Error_cost=-output(1,m+1);
        % if (Error_cost<-0.98)
            % break;
        % end
        % --- back-propagation parameters refresh ---
        [kernel_c1,kernel_f1,weight_f1,weight_output,bias_c1,bias_f1]=...
        CNN_upweight(yita,Error_cost,m,train_data,...
                     state_c1,state_s1,...
                     state_f1,state_f1_temp,...
                     output,...
                     kernel_c1,kernel_f1,weight_f1,...
                     weight_output,bias_c1,bias_f1);

    end    
end
end
toc
time_for_train=toc;


% --- test section ---
disp('train steps completed,starting test procedure......');
count=0;  % number of accurate identified samples
tic
for n=1:num_sample_to_get_test
    for m=0:9
        % samples for test
        test_data = reshape(preprocessed_test_data(m*num_sample_to_get_test+n,:),[28,28]);
        
        % train_data = get_average(train_data);
        % forward-propagation
        % convolution layer 1
        for k=1:layer_c1_num
            state_c1(:,:,k) = convolution(test_data,kernel_c1(:,:,k));
            % goto the reactifier layer tanh()
            state_c1(:,:,k) = tanh(state_c1(:,:,k)+bias_c1(1,k));
            % pooling layer-1
            state_s1(:,:,k) = pooling(state_c1(:,:,k),pooling_a);
        end
        % full-connected layer
        [state_f1_pre,state_f1_temp] = convolution_f1(state_s1,kernel_f1,weight_f1);
        % recitifier function layer
        for nn=1:layer_f1_num
            state_f1(1,nn) = tanh(state_f1_pre(:,:,nn)+bias_f1(1,nn));
        end
        % softmax layer
        for nn=1:layer_output_num
            output(1,nn) = exp(state_f1*weight_output(:,nn))/sum(exp(state_f1*weight_output));
        end
        [p,classify] = max(output);
        if (classify==m+1)
            count=count+1;
        end
        fprintf('real num: %d  model output: %d  confidence level: %d \n',m,classify-1,p);
    end
end
toc
time_for_test=toc;
fprintf('time for train: %d seconds time for test: %d seconds',time_for_train,time_for_test);
model_accuracy=count/num_sample_to_get_test;
fprintf('the total accuracy of this model on test data set is %d',model_accuracy);


