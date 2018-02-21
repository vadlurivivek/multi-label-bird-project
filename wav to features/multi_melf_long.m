clc; clear; close all;
allFiles = dir('../multi_data/multi_test_data_small_3');
dirFlags = [allFiles.isdir];
listFiles = allFiles(dirFlags);
len = length(listFiles);

h = waitbar(0,'Please wait...');
for i = 3:len
    
    inPath = '../multi_data/multi_test_data_small_3';
    lsFiles = dir([inPath '/' listFiles(i).name '/' '*.wav']);
    
    for j = 1:length(lsFiles)
        [y, Fs] = audioread([inPath '/' listFiles(i).name '/' lsFiles(j).name]);
        frameSize = ceil(20e-3*Fs);
        frameShift = ceil(10e-3*Fs);
        
        % For MFCC features
        mfcc=melcepst(y,Fs,'0dD',12,floor(3*log(Fs)),frameSize,frameShift);
        
        
        %%%%%%%%%%%%%%%%%%%%%%% Mean normalization %%%%%%%%%%%%%%%%%%%%%%%
        mfcc = mfcc - repmat(mean(mfcc), size(mfcc,1), 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~,nm,~] = fileparts(lsFiles(j).name);
        outPath = ['../multi_test_bird_mfccs_small_3/',listFiles(i).name,'/'];
        if ~isdir(outPath)
            mkdir(outPath);
        end
        dlmwrite([outPath,nm,'.mfcc'],mfcc,' '); 
        %dlmwrite(['../bird_features_gd/',listFiles(i).name,'/',nm,'.gd'],...
        %   tau,' '); 
       % dlmwrite(['./',listFiles(i).name,'/',nm,'.mfcc'],...
            %mfcc,' ');
    end
    
    waitbar(i/len,h)
    
   
end
close(h)
        
