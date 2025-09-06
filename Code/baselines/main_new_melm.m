%多隐藏层Elm加鹦鹉优化算法

%% 多标签分类      DA-ELM   穿插GWO、PSO和BOA
clear all
clc
%% 数据预处理
%rng(42);
% 定义任务类型
REGRESSION=0;
CLASSIFIER=1;                             %  
Elm_Type=1;
ActivationFunction='sig';           % sig 函数训练效果最好（仅做了隐含神经元个数为28的实验）             DA 训练集准确率：0.97659  测试集准确率：0.96406  总准确率：0.97346
                                  %  GWO  训练集准确率：0.98168  测试集准确率：0.96619  总准确率：0.97781
%% II. 训练集/测试集产生
%%
% 1. 导入数据
%load 8_4_node14_17000.mat
abc = randperm(17000);                        %%%%% 打乱数据
Pn_train = features(abc(1:11900),:);
Pn_test = features(abc(11901:end),:);
T_train = labels(abc(1:11900),:);
T_test = labels(abc(11901:end),:);



P = Pn_train';%训练集特征
TV.P = Pn_test';%测试集特征
T = T_train';%训练集分类
TV.T = T_test';%测试集分类
R = size(P,1);%特征数104；
%inputnum=size(P,2);%数据的数目1860；
S = size(T,1);%标签数28
% 定义优化问题的参数
SearchAgents=2;
Max_iterations=5;
%lb = [0.001,40];%下边界         %%%%%%%%%     14--0.001和40      57--0.001和40
% ub = [35,200];%上边界  
lowerbound=2;
upperbound=250;


dim = 3;
fobj= @(x)melm_function(x,Pn_train,T_train);
                          

[Alpha_score,Alpha_pos,Convergence_curve] = GWO(SearchAgents, Max_iterations,lowerbound, upperbound, dim, fobj)
[fitness,N_ae1,N_ae2,N_ae3,LW] = melm_function(Alpha_pos,Pn_train,T_train);   %获取优化后的相关参数  3      


IW1=2*rand(N_ae1,R)-1;
B1=rand(N_ae1,1);
IW2=2*rand(N_ae2, N_ae1)-1;
B2=rand(N_ae2,1);
IW3=2*rand(N_ae3,N_ae2)-1;
B3=rand(N_ae3,1);

NumberofTrainingData=size(P,2);        % 训练数据中分类对象个数
NumberofTestingData=size(TV.P,2);      % 测试数据中分类对象个数
NumberofInputNeurons=size(P,1);        % 神经网络输入个数，训练数据的属性个数

T=2*T-1;                              %%%%%%%%%%% 期望输出
% TV.T=2*TV.T-1;

%BiasofHiddenNeurons=B;    % 连接偏重在[0,1]之间

% 第一层的前向传播
tempH1 = IW1 * P + repmat(B1, 1, NumberofTrainingData);
H1 = 1 ./ (1 + exp(-tempH1)); 

% 第二层的前向传播
tempH2 = IW2 * H1 + repmat(B2, 1, NumberofTrainingData);
H2 = 1 ./ (1 + exp(-tempH2)); 

% 第三层的前向传播
tempH3 = IW3 * H2 + repmat(B3, 1, NumberofTrainingData);
H3 = 1 ./ (1 + exp(-tempH3));

% 使用最后一层的激活输出和伪逆计算输出权重
OutputWeight = pinv(H3') * T';


% 对于测试集的前向传播
tempH_test1 = IW1 * TV.P + repmat(B1, 1, NumberofTestingData); % 第一层
H_test1 = 1 ./ (1 + exp(-tempH_test1)); % 激活函数

tempH_test2 = IW2 * H_test1 + repmat(B2, 1, NumberofTestingData); % 第二层
H_test2 = 1 ./ (1 + exp(-tempH_test2)); % 激活函数

tempH_test3 = IW3 * H_test2 + repmat(B3, 1, NumberofTestingData); % 第三层
H_test3 = 1 ./ (1 + exp(-tempH_test3)); % 激活函数

Y =(H3' * OutputWeight)';  % Y为训练数据输出（列向量） 
TY = (H_test3' * OutputWeight)'; % 测试数据的输出   %   TY: 测试数据的输出


for t=1:28
 for i = 1:size(Y,2)
        if Y(t,i) > 0
            temp_y = 1;
        else
            temp_y = 0;
        end
        y_predict_train(t,i) = temp_y;
 end

  for j = 1:size(TY,2)
        if TY(t,j) > 0
            temp_y = 1;
        else
            temp_y = 0;
        end
        y_predict_test(t,j) = temp_y;
  end


  
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    T = T_train';                      % 与原数据对比
    TV.T = T_test';
    
    for i = 1 : size(T, 2)
        if y_predict_train(t,i)~=T(t,i)
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    
    for i = 1 : size(TV.T,2)
        if y_predict_test(t,i)~=TV.T(t,i)
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end

    TrainingAccuracy(t)=1-MissClassificationRate_Training/size(T,2); % 训练集分类正确率
    TestingAccuracy(t)=1-MissClassificationRate_Testing/size(TV.T,2);  % 测试集分类正确率
end

y_predict = cat(1,y_predict_train', y_predict_test');                  % cat 将两个矩阵合并到一起

A = 0; B = 0;
for i= 1 : 28
    A = TrainingAccuracy(i) + A;
    B = TestingAccuracy(i) + B;
    a = A/28;   b = B/28;
end
    c = (a*1736+ b*744)/2480;
disp(['训练集准确率：' num2str(a)]);
disp(['测试集准确率：' num2str(b)]);
disp(['总准确率：' num2str(c)]);
%% 评价结果
TY=y_predict_test;
TP=0;FN=0;FP=0;TN=0;   % 真正例 假负例 假正例 真负例
for i=1:size(TY,1)
    for t=1:size(TY,2)
         if (TY(i,t)==TV.T(i,t))&&(TV.T(i,t)==1)
             TP=TP+1;
         elseif (TY(i,t)~=TV.T(i,t))&&(TV.T(i,t)==1)
             FN=FN+1;
           elseif (TY(i,t)~=TV.T(i,t))&&(TV.T(i,t)==0)  
             FP=FP+1;
             elseif (TY(i,t)==TV.T(i,t))&&(TV.T(i,t)==0)  
                 TN=TN+1;
         end
    end
end

Accuracy=(TP+TN)/(TP+FN+FP+TN);                   % 准确率
Precision=TP/(TP+FP);                             % 精确率
Recall=TP/(TP+FN);                                % 召回率
F1=2*Precision*Recall/(Precision+Recall);         % F1值

FPR=FP/(FP+TN);                                   % ROC曲线横坐标
TPR=TP/(TP+FN);                                   % ROC曲线纵坐标

disp(['真正例：' num2str(TP)]);
disp(['假负例：' num2str(FN)]);
disp(['假正例：' num2str(FP)]);
disp(['真负例：' num2str(TN)]);
disp(['精度：' num2str(Precision)]);
disp(['召回率：' num2str(Recall)]);
disp(['F1值：' num2str(F1)]);