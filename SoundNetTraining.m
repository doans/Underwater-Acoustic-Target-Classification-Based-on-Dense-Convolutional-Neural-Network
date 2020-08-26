load UATC-DenseNet.mat
Classes = categorical(["Noise","T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",...
    "T09", "T10","T11"]);
% Download dataset via following link:
% https://o365kumoh-my.sharepoint.com/personal/thienht_office_kumoh_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fthienht%5Foffice%5Fkumoh%5Fac%5Fkr%2FDocuments%2FUnderwater%20Acoustic%20Target%20Classification%2FDataset%2Ezip&parent=%2Fpersonal%2Fthienht%5Foffice%5Fkumoh%5Fac%5Fkr%2FDocuments%2FUnderwater%20Acoustic%20Target%20Classification&originalPath=aHR0cHM6Ly9vMzY1a3Vtb2gtbXkuc2hhcmVwb2ludC5jb20vOnU6L2cvcGVyc29uYWwvdGhpZW5odF9vZmZpY2Vfa3Vtb2hfYWNfa3IvRWRvX2UyX3lydnhQdUdnR1REWlZHU1FCOE5qeDZyelV3dVB4WkFTeTRoUkdkUT9ydGltZT16UFZBTG1aSjJFZw
imds = imageDatastore('NewDataset_03','IncludeSubfolders',true,...
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