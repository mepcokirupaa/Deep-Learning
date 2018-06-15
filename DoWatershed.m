function W = DoWatershed(I)
    %% I is a logical mask (i.e., im2bw)
    d = -bwdist(~I);%distance between a pixel and nearest nonzero pixel    
    l = watershed(d); %to avoid cell overlapping 
    W = I; 
    W(l == 0) = 0;% to mark separation by white color
    figure,imshow(W), title('After Watershed');
    bin1 = imfill(W,'holes'); %filling holes in W
    figure,imshowpair(W,bin1,'montage');
%     figure,imshow(bin3);
    bin2 = imopen(bin1,strel('disk',7,8)); %converts to disc shape. radius -7, height-8
%     figure,imshow(bin3);
    bin3 = bwareaopen(bin2, 800);  %if <800 pixels remove the small objects 
	
	%now bin3 contains the final resultant image
%     figure,imshow(bin3);
    figure,imshowpair(bin2,bin3,'montage');
    
	%to write the converted image into a destination folder
	
	destinationFolder = 'D:\subjectsHandled\B.E-Report\2017-18\VIII-Sem-Project\Kirupa-Keerthana-kavya\Project\Biomarkers for the Identification of Acute Leukemia (April '18)\Code\Udata';
	if ~exist(destinationFolder, 'dir')
		mkdir(destinationFolder);
	end
	% Strip off extenstion from input file
	%[sourceFolder, baseFileNameNoExtenstion, ext] = fileparts(filename);
	% Create jpeg filename.  Don't use jpeg format for image analysis!
	%outputBaseName = [baseFileNameNoExtenstion, '.JPG'];
	%fullDestinationFileName = fullfile(destinationFolder, outputBaseName);
	% Write the jpg file.  This will convert whatever format you started with to the hated jpg format.
	%imwrite(I, fullDestinationFileName);


end