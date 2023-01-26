%%
clc; clear; close all;

%% data import

load Indian_pines.mat;


%% normalization 

hcube = hypercube('indian_pines.dat');
data = hcube.DataCube;

x_range = 1:100;
y_range = 1:50;
z_range = 1:25; 
reference = data(x_range,y_range,z_range);
lower_bound = min(reference(:));
upper_bound = max(reference(:));

rescaled_data = rescale(data,lower_bound,upper_bound);
hcube = hypercube(rescaled_data,hcube.Wavelength);
%% 3- Suitable bands
endmembers = fippi(hcube,9);
newhcube= selectBands(hcube,endmembers,'NumberOfBands',4 ); 
coloredImage = colorize(newhcube );

%%4
load Indian_pines_gt.mat;
figure; title(' Ground Truth'); imagesc( indian_pines_gt);

%%5
histogram(indian_pines);

%% 6 
reducedDataCube3 = hyperpca(hcube,3);
reducedDataCube5 = hyperpca(hcube,5);
reducedDataCube10 = hyperpca(hcube,10);
reducedDataCube20 = hyperpca(hcube,20);
reducedDataCube25 = hyperpca(hcube,25);
%% 7 false color

FC(:,:,1)= indian_pines(:,:,2)/ norm(indian_pines(:,:,2));
FC(:,:,2)= indian_pines(:,:,20)/ norm(indian_pines(:,:,20));
FC(:,:,3)= indian_pines(:,:,45)/ norm(indian_pines(:,:,45));
figure;title('The false folor image'); imshow(FC);
%% 8 

Y = pdist( reducedDataCube3);

Y = squareform( Y );
figure; imagesc(Y);

%%9 SVM
% grid search + cross-validation
kernel='-t 2 '; %rbf kernel
base = 10;   
bestcv = 0; % for grid search
for ig = 1 : 6 % gamma parameters values (rows from 0.001 to 100)
    logbaseg = (ig - 4);
    for ic = 1 : 6   % c parameters values (columns from 0.1 to 10000) 
        logbasec = (ic - 2);
        % cross-validation (5 fold= -v 5)
        cmd = [kernel, '-v 5 -c ', num2str(base^logbasec), ' -g ', num2str(base^logbaseg)];
        cv = svmtrain(trainLabels', D_train, cmd);
        CV(ig,ic) = cv;   % grid searh matrix
        if(cv >= bestcv) 
            bestcv = cv; bestc = base^logbasec; bestg = base^logbaseg;
        end
    end
end
    
% training with optimised parameter gamma and c
cmd = [kernel,' -c ' num2str(bestc) ' -g ' num2str(bestg)];
model = svmtrain(trainLabels', D_train, cmd);
    
% testing
[predictedLabels, accuracy, ~] = svmpredict(testLabels', D_test, model);
   
% showing results
oa = accuracy(1); % result

%% plot actual classes vs predicted classes
figure; hold on; plot( testLabels, 'ob' ); plot( predictedLabels, '*r');
legend('Actual Labels', 'Predicted Labels', 'Location', 'best');
title('Actual Labels vs Predicted Labels');
xlabel('Testing Samples'); ylabel('Classes');

%% confusion matrix
CM = confusionmat(testLabels, predictedLabels);
figure; confusionchart( CM );

%% metrics
overallAccuracy = sum( diag(CM) ) / sum( CM(:)  ) * 100;

numberOfClasses = length( unique( testLabels ) );
% average accuracy
AA = zeros(numberOfClasses, 1);
for i = 1 : numberOfClasses
   AA(i) = CM(i, i) / sum( CM(i, :) ) * 100;  
end
averageAccuracy = mean( AA ); 





