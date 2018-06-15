srcFiles = dir('D:\subjectsHandled\B.E-Report\2017-18\VIII-Sem-Project\Kirupa-Keerthana-kavya\Project\ALLDataSet\ALL_IDB1\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('D:\subjectsHandled\B.E-Report\2017-18\VIII-Sem-Project\Kirupa-Keerthana-kavya\Project\ALLDataSet\ALL_IDB1\',srcFiles(i).name);
    I = imread(filename);
	MKmeans(I);
    %I= imresize(I,[150 ,150],'Antialiasing',false); %don't resize
    figure, imshow(I);

% Create destination filename
%destinationFolder = 'C:\Users\DELL\Documents\MATLAB\test1\Healthy';
%if ~exist(destinationFolder, 'dir')
% mkdir(destinationFolder);
%end
% Strip off extenstion from input file
%[sourceFolder, baseFileNameNoExtenstion, ext] = fileparts(filename);
% Create jpeg filename.  Don't use jpeg format for image analysis!%
%outputBaseName = [baseFileNameNoExtenstion, '.JPG'];
%fullDestinationFileName = fullfile(destinationFolder, outputBaseName);
% Write the jpg file.  This will convert whatever format you started with to the hated jpg format.
%imwrite(I, fullDestinationFileName);

end