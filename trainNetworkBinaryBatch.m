%==========================================================================
%TRAINNETWORKSTE Trains a network TO BE COMPLETELY BINARY USING A DERIVATIVE OF TANH FOR THE BACKPROPAGATION
%   @Input
%   X: Data for training
%   y: Labels for X
%   network: Network that should be trained
%   varargin:
%       'epochs'    Number of epochs you want to train
%       'alpha'     Alpha you want to use for training
%       'validationData'        Validation dataset
%       'validationDataOutput'  Labels for the validation dataset
%
%
%   @Output
%   trainedNetwork:
%   cost_log: Vector with the cost after each epoch
%   trainingSetAccuracy: Vector with the accuracy of the network after each epoch
%   validationSetAccuracy: Vector with the accuracy of the valdidation dataset (if given) after each epoch
%
% Algorithm for the Training
% Forward Propagation:
%   For each layer:
%     Binarize the weights
%     multiply the weights with the layer input
%     Batch Normalization
%     if there are not more layers left
%       Binarize the inputs
% Backward Propagation:
%   Compute the derivative of the cost function
%   For each layer:
%     Do the straight through estimator
%     Calculate the backward batch normalization
%     Update binary  activations and weights
% Update real value weights and learning rate and batchNorm parameter (thta)

function [trainedNetwork, cost_log, trainingSetAccuracy, validationSetAccuracy] = trainNetworkBinaryBatch (X, y, network, varargin)
  %default parameter
    defaultEpochs=100;
    defaultAlpha=0.0001;
    defaultValidationData=[];
    defaultValidationOutput=[];

    %Input Parser
    p = inputParser;
    p.FunctionName = 'trainNetworkSTE';
    addParameter(p,'epochs',defaultEpochs,@(x)validateattributes_with_return_value(x,{'numeric'},{'nonempty'}));
    addParameter(p,'alpha',defaultAlpha,@(x)validateattributes_with_return_value(x,{'numeric'},{'nonempty'}));
    addParameter(p,'validationData',defaultValidationData);
    addParameter(p,'validationDataOutput',defaultValidationOutput);
    p.parse(varargin{:});

    %Assign parsed input to the variables
    epochs = p.Results.epochs;
    alpha = p.Results.alpha;
    validationData = p.Results.validationData;
    validationOutput = p.Results.validationDataOutput;
    epsilon=0.001;
    gamma = 1;  %scaling factor
    beta = 0; % offset factor

    %Only do a validation if the user inserts a validation dataset with the
    %expected results
    doValidation = ~isempty(validationData) && ~isempty(validationOutput);

    % weights of the network with real values
    theta = network;
    % weights of the network with binary values
    theta_binary = network;

    %number of weight matrices and layers
    disp("number of weight matrices");
    numberOfThetas = length(theta)
    disp("number of layers");
    numberOfLayers = numberOfThetas +1

    %matrices for the Output of each Layer

    layer=cell(1,numberOfLayers); % set that will have arrays for each layer

    %this will be used later for the cost function
    m=size(X,1);
    cost_log=zeros(epochs,1); % set that saves costs of epochs
    trainingSetAccuracy=zeros(epochs,1); % array that saves training accuracy of epoch
    validationSetAccuracy=zeros(epochs,1);

     %assign the transposed input of the convoluted images to the input layer
    layer{1} = X';
    %Add offset to the first layer (input layer)
    layer{1}=[layer{1}; ones(1,size(layer{1},2))];

    %binarization of the real value weights, set the weights of the network to binary
    for j=1:numberOfThetas
      theta_binary{j} = deterministic_binarization(theta{j}); %binarization of the real value weights
    endfor

        %loop the carry out gradient descent iter times
    for i=1:epochs
        fprintf("Epoch %d/%d\r",i,epochs);
        %forward propagation to calculate output using sigmoid function

        %printf("Size of theta binary");
        %disp(size(theta_binary));


        %Activation function
        for j=1:numberOfThetas
            %By the forward calculation the offset neuron gets inserted into the activation function. This needs to be reveresed before the next layer is calculated
            layer{j}(end,:)=1;
            %layer{j+1} = deterministic_binarization(theta_binary{j} * layer{j}); %activation function (and gate?)
            v=[]
            %apply binary cross coreelation  layer{j} which are the input values with theta_binary{j} which are the binary weight vaues this is like multiplying and applying the activation function after applying binarization
            for g = 1:rows(theta_binary{j})
              %layer{j+1}(g)= deterministic_binarization(binaryCrossCorrelation(theta_binary{j}(g,:),layer{j}');)';
              v(g)=binaryCrossCorrelation(theta_binary{j}(g,:),layer{j}');
            endfor
            v = batchNormalization(layer{v},epsilon,gamma,delta);
            layer{j+1}=deterministic_binarization(v)';
            v=[];

        end

        %forward batch normalization
        for j=1:numberOfThetas
            theta{j} = batchNormalization(layer{j+1},epsilon,gamma,delta);   %Batch Normalization Forward Pass
            gamma=sqrt(var(layer{j+1}));
            delta=mean(layer{j+1});
        end





        % update the binary values of the weights
        for j=1:numberOfThetas
          theta_binary{j} = deterministic_binarization(theta{j}); %binarization of the real value weights
        endfor

        %back propagation to calculate error
        %%error=cell(1,numberOfLayers);
        %error{numberOfLayers} = y';
        %%error{numberOfLayers} = layer{numberOfLayers} - y'; %The error for the output layer is calculated outside of the for loop



        %%for j=numberOfThetas: -1 :2
        %%%%   error{j} = theta_binary{j}' * error {j+1}; % multiplicate matrices of output error and weights of the previous layer of network, whose weights are binarized
        %%end

        % disp("error size")
        % size(error)




        % derivative of sigmoid transfer(activation) function is output*(1-output)
        % Substract partial derivatives from theta
        % The error for a given neuron can be calculated as follows:
        % error = (output - expected) * transfer_derivative(output)  Where expected is the expected output value for the neuron, output is the output value for the neuron and transfer_derivative() calculates the slope of the neuron’s output value, as shown above. This error calculation is used for neurons in the output layer.
        % The back-propagated error signal is accumulated and then used to determine the error for the neuron in the hidden layer, as follows:
        % error = (weight_k * error_j) * transfer_derivative(output)
        % disp("size theta")
        %size(theta{1})
        for j=1:numberOfThetas

            if (j < numberOfThetas)
            %disp("subtractor")
            %theta{j} = theta{j} - alpha * ((error{j+1} .* layer{j+1} .* (1-layer{j+1})) * layer{j}'); %original backpropagation without ste
            %subtractor = alpha * ((error{j+1} .* layer{j+1} .* (1-layer{j+1})) * layer{j}');
            [xhat, xmu, ivar, sqrtvar, var] = batchnorm_forward (batch, epsilon, gamma, beta)
            subtractor = alpha * (deterministic_binarization(layer{j+1})*identityOne(layer{j+1}) ); % new ste proposal
            %theta{j} = identityOne(theta{j} - alpha * (error{j+1})); % even newer ste proposal
            %subtractor =  alpha * (error{j+1} * layer{j}');
            %size(subtractor)
            %disp("theta{j}")
            %size(theta{j})
              theta{j} =(theta{j} -  subtractor); % updating weights with gradient descent: weight - learningrate*error*input and adding hard tanh to keep them in 1 : -1 range
             elseif(j=numberOfThetas)
              [dx, dgamma, dbeta] = batchnorm_backward (theta{j}, xhat, gamma, epsilon, xmu, ivar, sqrtvar, var)
              gradient_binary{-1} = dx*theta_binary{j};
              gradient_weight=dx*gradient_binary{-1}


      end

      for j=1:numberOfThetas
        # update batchnorm parameters
        [xhat, xmu, ivar, sqrtvar, var] = batchnorm_forward (theta{j}, epsilon, gamma, beta)
        # update weights between -1 and 1
        theta{j}=hardTanh(theta{j})
        gamma = alpha*gamma
      endfor



        %Calculate Mean Square Error for each epoch
        %Double sum because sum of a matrix creates a vector and sum of a
        %vector creates a double value.
        cost = 1/m * sum(sum(error{numberOfLayers}.^2));
        cost_log(i)=cost;

        %Calculate the accuracy of the trainings set
        trainingSetAccuracy(i)=calculateAccuracy(layer{numberOfLayers}, y');

        if(doValidation)
            prediction=networkPrediction(validationData, theta);
            validationSetAccuracy(i)=calculateAccuracy(prediction, validationOutput');
        end

    end

    trainedNetwork = theta;





endfunction
