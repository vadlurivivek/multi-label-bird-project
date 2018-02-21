clc; clear; close all;

listFiles = dir;
dirFlags = [listFiles.isdir];
listFiles = listFiles(dirFlags);
len = length(listFiles);

for i = 3:len
    
    inPath = ['./' listFiles(i).name];
    lsFiles = dir([inPath '/*.wav']);
    
    for j = 1:length(lsFiles)
        [y, Fs] = audioread([inPath '/' lsFiles(j).name]);
        frameSize = ceil(20e-3*Fs);
        frameShift = ceil(10e-3*Fs);
        
        % For log mel filterbank coefficients
%       [logmel,~] = melcepst(y,Fs,'0dD',12,floor(3*log(Fs)),frameSize,frameShift);
        [logmel,~] = melcepst(y,Fs,'0dD',12,48,frameSize,frameShift);
        logmel = logmel';
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~,nm,~] = fileparts(lsFiles(j).name);
        outPath = ['../bird_melfilter_48/',listFiles(i).name,'/'];
        if ~isdir(outPath)
            mkdir(outPath);
        end
        dlmwrite([outPath,nm,'.mel'],logmel,' '); 
    end
end
        
