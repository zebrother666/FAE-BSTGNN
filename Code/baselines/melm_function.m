%三层隐藏层elm,与main_new_melm配套

%% 只优化偏置，输入权重随机生成          多标签
function [fitness,N_ae1,N_ae2,N_ae3,OutputWeight]= melm_function(x,Pn_train,T_train)

% load 14-2480.mat
% 
% Pn_train = features(1:1860,:);
% T_train = classes(1:1860,:);
% Pn_test =  features(1861:2480,:);
% T_test = classes(1861:2480,:);

P = Pn_train';
T = T_train';

R = size(P,1);
S = size(T,1);


N_ae1=round(x(1));
N_ae2=round(x(2));
N_ae3=round(x(3));

% 参数解包

IW1=2*rand(N_ae1,R)-1;
B1=rand(N_ae1,1);
IW2=2*rand(N_ae2, N_ae1)-1;
B2=rand(N_ae2,1);
IW3=2*rand(N_ae3,N_ae2)-1;
B3=rand(N_ae3,1);



% 第一层ELM处理
H1 = 1 ./ (1 + exp(-(IW1*P + B1)));

% 第二层ELM处理
H2 = 1 ./ (1 + exp(-(IW2*H1 + B2)));

% 第三层ELM处理
H3 = 1 ./ (1 + exp(-(IW3*H2 + B3)));

% 计算输出权重
OutputWeight = pinv(H3') * T';

% 生成训练集预测
Y = (H3' * OutputWeight)'; 

% 评估性能
k=zeros(1, size(T_train, 2));
T = T_train';
%14节点的
for t=1:28
    for i = 1:size(Y,2)
        if Y(t,i) > 0
            temp_y = 1;
        else
            temp_y = 0;
        end
        y_predict_train(t,i) = temp_y;
    end
  
    k(t)=length(find(y_predict_train(t,:) ~= T(t,:)));

end
TY=y_predict_train;
TP=0;FN=0;FP=0;TN=0;   % 真正例 假负例 假正例 真负例
for i=1:size(TY,1)
    for t=1:size(TY,2)
         if (TY(i,t)==T(i,t))&&(T(i,t)==1)
             TP=TP+1;
         elseif (TY(i,t)~=T(i,t))&&(T(i,t)==1)
             FN=FN+1;
           elseif (TY(i,t)~=T(i,t))&&(T(i,t)==0)  
             FP=FP+1;
             elseif (TY(i,t)==T(i,t))&&(T(i,t)==0)  
                 TN=TN+1;
         end
    end
end

Accuracy=(TP+TN)/(TP+FN+FP+TN);                   % 准确率
fitness=1-Accuracy;


%fitness=sum(k)/(28*2480);
end














% 
% %% 只优化偏置，输入权重随机生成          多标签
% function [fitness,IW,B,OutputWeight] = f2(x,N)
% % NumberofHiddenNeurons=10;
% load 14-2480.mat
% 
% Pn_train = features(1:1860,:);
% T_train = classes(1:1860,:);
% Pn_test =  features(1861:2480,:);
% T_test = classes(1861:2480,:);
% P = Pn_train';
% TV.P = Pn_test';
% R = size(P,1);
% IW = x(1:N*R);
% 
% %  t=6;
%     T = T_train';
%  
%     TV.T = T_test';
%     
% 
% S = size(T,1);
% B = x(N*R+1:end);
% % B = x(N*R+1:N*R+N*S);
% IW = reshape(IW,[N,R]);
% B = reshape(B,N,1);
% 
% NumberofTrainingData=size(P,2);        % 训练数据中分类对象个数
% NumberofTestingData=size(TV.P,2);      % 测试数据中分类对象个数
% NumberofInputNeurons=size(P,1);        % 神经网络输入个数，训练数据的属性个数
% 
% T=2*T-1;                              %%%%%%%%%%% 期望输出
% TV.T=2*TV.T-1;
% 
% BiasofHiddenNeurons=B;    % 连接偏重在[0,1]之间
% 
% tempH=IW*P; % 不同对象的属性*权重
% %clear P; % 释放空间 
% ind=ones(1,NumberofTrainingData);     % 训练集中分类对象的个数
% BiasMatrix=BiasofHiddenNeurons(:,ind);% 扩展BiasMatrix矩阵大小与H匹配 
% tempH=tempH+BiasMatrix;               % 加上偏差的最终隐层输入
% H = 1 ./ (1 + exp(-tempH));
% OutputWeight=pinv(H') * T';
% 
% % tempH_test=IW*TV.P;% 测试的输入
% % ind=ones(1,NumberofTestingData);
% % % BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
% % BiasMatrix=BiasofHiddenNeurons(:,ind); % 扩展BiasMatrix矩阵大小与H匹配 
% % tempH_test=tempH_test + BiasMatrix;% 加上偏差的最终隐层输入
% % H_test = 1 ./ (1 + exp(-tempH_test));
% 
% 
% Y =(H' * OutputWeight)';                            % Y为训练数据输出（列向量） 
% % TY=(H_test' * OutputWeight)';                       % TY: 测试数据的输出
%   
%   
% k=zeros(1,28);
% % g=zeros(1,28);
% T = T_train';
% % TV.T = T_test';
% for t=1:28
%     for i = 1:size(Y,2)
%         if Y(t,i) > 0
%             temp_y = 1;
%         else
%             temp_y = 0;
%         end
%         y_predict_train(t,i) = temp_y;
%     end
% 
% %   for j = 1:620
% %         if TY(t,j) > 0
% %             temp_y = 1;
% %         else
% %             temp_y = 0;
% %         end
% %         y_predict_test(t,j) = temp_y;
% %   end
%   
%     k(t)=length(find(y_predict_train(t,:) ~= T(t,:)));
% %     g(t)=length(find(y_predict_test(t,:) ~= TV.T(t,:)));
% end
% 
% %      fitness=(sum(k)+sum(g))/(56*2480);
%      fitness=sum(k)/(28*2480);
% end




