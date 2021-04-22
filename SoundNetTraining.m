load UATC-DenseNet.mat
Classes = categorical(["Noise","T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",...
    "T09", "T10","T11"]);
% Download dataset via following link:
% https://mega.nz/folder/q1tgjJCA#GO8yhqz0s5Jm1pf18Vz9qA
imds = imageDatastore('Dataset','IncludeSubfolders',true,...
    'LabelSource','foldernames','FileExtensions',{'.mat'});
% Split dataset into 2 sub-sets of training 70% and testing 30%
[imdsTrain,imdsTest,other] = splitEachLabel(imds,0.7,0.3,'randomized');
% Read data of training
imdsTrain.Labels = categorical(imdsTrain.Labels);
imdsTrain.ReadFcn = @readFcnMatFile;
% Read data of testing
imdsTest.Labels = categorical(imdsTest.Labels);
imdsTest.ReadFcn = @readFcnMatFile;

%% Network generation
% Set up training option
batchSize   = 64;
ValFre      = fix(length(imdsTrain.Files)/batchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',batchSize, ...
    'MaxEpochs',40, ...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.1,...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',ValFre, ...
    'ValidationPatience',Inf, ...
    'L2Regularization',0.0001,...
    'Verbose',true ,...
    'VerboseFrequency',ValFre,...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');


%% Training
tic
trainNow = true;
if trainNow == true
    trainedNet = trainNetwork(imdsTrain,lgraph_1,options);
    fprintf('%s - Training the network\n', datestr(toc/86400,'HH:MM:SS'))
    timeFormat = 'yyyy-mm-dd-HH-MM-SS';
    now = datetime('now');
    Trained_file = sprintf('SoundTrainedNet_new%s.mat', datestr(now,timeFormat));
    save(Trained_file, 'trainedNet', 'lgraph_1');
else
    %   load trainedModulationClassificationNetwork
end

%% Testing
% Measure the accuracy on the test set
YPredic = classify(trainedNet,imdsTest,'MiniBatchSize',64,'ExecutionEnvironment','gpu');
YTest = imdsTest.Labels;
% Compared the prediction vs the groudtruth
accuracy = sum(YPredic == YTest)/numel(YTest)*100;

%% Plot the result in confusion matrix
conf = confusionmat(YTest,YPredic);
confmat = conf(25:end,25:end);

figure
cm = confusionchart(conf(:,:,1),Classes);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.FontName = 'Times New Roman';
cm.FontColor =[0 0 0];
cm.FontSize = 12;
colorbar = [0 0 1];
cm.DiagonalColor = colorbar;
cm.Parent.Position = [cm.Parent.Position(1:2) 740 424];
cm.OffDiagonalColor = colorbar;
