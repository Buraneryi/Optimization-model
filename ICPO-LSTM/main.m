%%
warning off          
close all               
clear                 
clc                     

%%  
result = xlsread('数据集.xlsx');

%%  
num_samples = length(result); 
kim = 1;                       
zim =  1;                      

%%  
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1), 1, kim), result(i + kim + zim - 1)];
end

%%  
outdim = 1;                                
num_size = 0.7;                             
num_train_s = round(num_size * num_samples); 
f_ = size(res, 2) - outdim;                 

%%  
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end
%%  
ObjFcn = @CostFunction;

%% 
SearchAgents = 6; 
Max_iterations = 10 ;  
lowerbound = [1e-8 0.0001 2 ];
upperbound = [1e-1 0.1 100 ];
dimension = 3;

%% 
[Best_score,Best_pos,Convergence_curve]=ICPO(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,ObjFcn);

%%  
NumOfUnits       =round( Best_pos(1,3));       
InitialLearnRate = Best_pos(1,2); 
L2Regularization = Best_pos(1,1); 

%%  
layers = [ ...
    sequenceInputLayer(size(P_train,1))                
    lstmLayer(NumOfUnits)                       
    reluLayer                                       
    fullyConnectedLayer(outdim)                     
    regressionLayer];                               

%%  
options = trainingOptions('adam', ...                 
    'MaxEpochs', 1200, ...                         
    'GradientThreshold', 1, ...                       
    'InitialLearnRate', InitialLearnRate, ...        
    'LearnRateSchedule', 'piecewise', ...             
    'LearnRateDropPeriod', 800, ...                  
    'LearnRateDropFactor',0.1, ...                    
    'L2Regularization', L2Regularization, ...       
    'Verbose', 0, ...                                 
    'Plots', 'training-progress');                    

%%  
net = trainNetwork(vp_train, vt_train, layers, options);

%% 
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%%  
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  
T_sim1 = cell2mat(T_sim1);
T_sim2 = cell2mat(T_sim2);
T_sim1 = double(T_sim1');
T_sim2 = double(T_sim2');


%%  
figure
plot(Convergence_curve,'b-', 'LineWidth', 1.5);
title('CPO-LSTM', 'FontSize', 10);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
grid off
set(gcf,'color','w')

%%  MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

%% 
figure;
plotregression(T_test,T_sim2,'回归图');
set(gcf,'color','w')
figure;
ploterrhist(T_test-T_sim2,'误差直方图');
set(gcf,'color','w')

%%  
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%  
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%% 
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;

%% 
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;
SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;

%% 
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));

%% 
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));

%% 
figure
plot(1:M,T_train,'r-',1:M,T_sim1,'b-','LineWidth',1.5)
legend('真实值','CPO-LSTM预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
set(gcf,'color','w')
%% 
figure
plot(1:N,T_test,'r-',1:N,T_sim2,'b-','LineWidth',1.5)
legend('真实值','CPO-LSTM预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
set(gcf,'color','w')
%% 
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('测试集样本编号')
ylabel('预测误差')
title('测试集预测误差')
grid on;
legend('CPO-LSTM预测输出误差')
set(gcf,'color','w')
%% 
%% 
figure
plot(T_train,T_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% 
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 
disp('-----------------------误差计算--------------------------')
disp('预测集的评价结果如下所示：')
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
grid