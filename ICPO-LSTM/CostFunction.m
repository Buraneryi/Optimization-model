function valError = CostFunction(optVars)
%%
L2Regularization =abs(optVars(1)); 
InitialLearnRate=abs(optVars(2)); 
NumOfUnits = abs(round(optVars(3))); 
%% 
vp_train = evalin('base', 'vp_train');
vt_train = evalin('base', 'vt_train');

%% 
inputSize    = size(vp_train{1}, 1);
numResponses = size(vt_train{1}, 1);

%% 
opt.layers = [ ...
    sequenceInputLayer(inputSize)       

    lstmLayer(NumOfUnits)             
    reluLayer                           

    fullyConnectedLayer(numResponses)   

    regressionLayer];

%%  
opt.options = trainingOptions('adam', ...            
    'MaxEpochs', 1200, ...                          
    'GradientThreshold', 1, ...                    
    'InitialLearnRate', InitialLearnRate, ...      
    'LearnRateSchedule', 'piecewise', ...             
    'LearnRateDropPeriod', 800, ...                
    'LearnRateDropFactor',0.1, ...                  
    'L2Regularization', L2Regularization, ...      
    'Verbose', 0, ...                                
    'Plots', 'none');                              

%% 
net = trainNetwork(vp_train, vt_train, opt.layers, opt.options);

%%  
t_sim1 = predict(net, vp_train); 

%%  
valError = sqrt(double(mse(vt_train, t_sim1)));

end