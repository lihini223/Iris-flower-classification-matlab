load iris_dataset irisInputs irisTargets
inputs = irisInputs;
targets = irisTargets;


    hiddenLayerSize = 5;
    accuracySum = 0;

    net =feedforwardnet(hiddenLayerSize);
    net.divideParam.trainRatio = 60/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 40/100;

    for index = 0:3
        for iteration = 0:9
            [net, tr] = train(net,inputs,targets);
            outputs = net(inputs);
            e = gsubtract(targets,outputs);

            performance = perform(net,targets,outputs);

            tind = vec2ind(targets);
            yind = vec2ind(outputs);
            percentErrors = sum(tind ~= yind)/numel(tind);

            acc = 100 * (1 - percentErrors);


            fprintf('Hidden Layer Size =%d  Accuracy = %.3f%% \n',hiddenLayerSize, acc);
            accuracySum = accuracySum + acc;
       %     view(net);
            plotperform(tr);
        
            testX = inputs(:,tr.testInd);
            testT = targets(:,tr.testInd);

            testY = net(testX);
            testIndices = vec2ind(testY);
%            plotconfusion(testT,testY);
             [c,cm] = confusion(testT,testY);
        
            fprintf('Correct Classification Percentage  : %f%%\n', 100*(1-c));
            fprintf('Incorrect Classification Percentage: %f%%\n', 100*c);
            fprintf('-----------------------------------------------------\n');
        end 
        
        average = accuracySum/10;
        fprintf('Hidden Layer %d --> Average Accuracy:%d \n', average);
        hiddenLayerSize = hiddenLayerSize + 5;
        fprintf('------------------Starting---------------------\n');
        
    end







