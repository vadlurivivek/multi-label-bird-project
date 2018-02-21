clc; clear; close all;

listFiles = dir;
dirFlags = [listFiles.isdir];
listFiles = listFiles(dirFlags);
len = length(listFiles);

h = waitbar(0,'Please wait...');

for i = 3:len

    inPath = ['./' listFiles(i).name];
    lsFiles = dir([inPath '/*.wav']);

    parfor j = 1:length(lsFiles)
        [y, Fs] = audioread([inPath '/' lsFiles(j).name]);
%         y = resample(y,160,441);
%         Fs = 16000;
        frameSize = ceil(20e-3*Fs);
%         frameShift = ceil(10e-3*Fs);
        frameShift = frameSize;
        numberOfFrames = length(y)/frameShift;
        numberOfFrames = floor(numberOfFrames);
        numberOfFrames = numberOfFrames - 1;
        raw = zeros(numberOfFrames, frameSize);
        startIndex = 1;
        for it=1:numberOfFrames
            raw(it,:) = y(startIndex:(startIndex+frameSize-1))';
            startIndex = startIndex + frameShift - 1;
        end

%         For MFCC features
%          mfcc=melcepst(y,Fs,'0dD',12,floor(3*log(Fs)),frameSize,frameShift);
        % For group delay features
%         [frames,~,~] = enframe(y,hamming(frameSize),frameShift);
%         tau = extract_lpGdelayVec_diff(frames,5,2048);
%         tau = dct(tau',13);
%         tau=tau';
%
%         %%%%%%%%%%%%%%% To compute delta and double delta %%%%%%%%%%%%%%%%
%
%         nf=size(tau,1);
%         nc=13;
%         vf=(4:-1:-4)/60;
%         af=(1:-1:-1)/2;
%         ww=ones(5,1);
%         cx=[tau(ww,:); tau; tau(nf*ww,:)];
%         vx=reshape(filter(vf,1,cx(:)),nf+10,nc);
%         vx(1:8,:)=[];
%         ax=reshape(filter(af,1,vx(:)),nf+2,nc);
%         ax(1:2,:)=[];
%         vx([1 nf+2],:)=[];
%         tau=[tau vx ax]; % for delta, remove vx if not necessary
%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%% Mean normalization %%%%%%%%%%%%%%%%%%%%%%%
        % tau = tau - repmat(mean(tau), nf, 1);
%         mfcc = mfcc - repmat(mean(mfcc), size(mfcc,1), 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [~,nm,~] = fileparts(lsFiles(j).name);
        % dlmwrite(['../bird_features_gd/',listFiles(i).name,'/',nm,'.gd'],...
            %tau,' ');
%         dlmwrite(['../mfcc_features_all/',listFiles(i).name,'/',nm,'.mfcc'],...
%             mfcc,' ');
          mkdir(['../raw_data_all/test_no_overlap/',listFiles(i).name]);
          mkdir(['../raw_data_all/train_no_overlap/',listFiles(i).name]);
          dlmwrite(['../raw_data_all/test_no_overlap/',listFiles(i).name,'/',nm,'.raw'],...
             raw,' ');
    end

    waitbar(i/len,h)
end
close(h)
