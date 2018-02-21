allFiles = dir('./test_small/');
dirFlags = [allFiles.isdir];
listFiles = allFiles(dirFlags);
len = length(listFiles);
outpath = './multi_test_data_small_3';

for i = 3:len
    inPath = './test_small/';
    lsFiles1 = dir([inPath '/' listFiles(i).name '/' '*.wav']);
    for m = 1:length(lsFiles1)
        for j = (i+1):len
            lsFiles2 = dir([inPath '/' listFiles(j).name '/' '*.wav']);
            for k = 1:length(lsFiles2)
               for q = (j+1):len
                    lsFiles3 = dir([inPath '/' listFiles(q).name '/' '*.wav']);
                    for b = 1:length(lsFiles3)
                        [x1, Fs1] = audioread(['./test_small/' listFiles(i).name '/' lsFiles1(m).name]);
                        [x2, Fs2] = audioread(['./test_small/' listFiles(j).name '/' lsFiles2(k).name]);
                        [x3, Fs3] = audioread(['./test_small/' listFiles(q).name '/' lsFiles3(b).name]);
                        min_len = min([length(x1), length(x2), length(x3)]);
                        y = x1(1:min_len) + x2(1:min_len) + x3(1:min_len);
                        file1_name = strsplit(lsFiles1(m).name, '.');
                        file2_name = strsplit(lsFiles2(k).name, '.');
                        file3_name = strsplit(lsFiles3(b).name, '.');
                        name_file = strcat(file1_name(1), '_', file2_name(1), '_', file3_name(1));
                        outpathfinal = [outpath '/' listFiles(i).name '_' listFiles(j).name '_' listFiles(q).name];
                        if ~isdir(outpathfinal)
                            mkdir(outpathfinal);
                        end
                        file_name = strcat(outpathfinal, '/', name_file, '.wav');
                        audiowrite(file_name{1,1}, y, Fs1);
                    end
               end  
            end
        end
    end
end



%   
%                 [x1, Fs1] = audioread(['./test_small/' listFiles(i).name '/' lsFiles1(m).name]);
%                 [x2, Fs2] = audioread(['./test_small/' listFiles(j).name '/' lsFiles2(k).name]);
%                 y = x1(1:min(length(x1), length(x2))) + x2(1:min(length(x1), length(x2)));
%                 file1_name = strsplit(lsFiles1(m).name, '.');
%                 file2_name = strsplit(lsFiles2(k).name, '.');
%                 name_file = strcat(file1_name(1), '_', file2_name(1), '_', listFiles(i).name, '_', listFiles(j).name);
%                 outpathfinal = [outpath '/' listFiles(i).name '_' listFiles(j).name];
%                 if ~isdir(outpathfinal)
%                     mkdir(outpathfinal);
%                 end
%                 file_name = strcat(outpathfinal, '/', name_file, '.wav');
%                 audiowrite(file_name{1,1}, y, Fs1);