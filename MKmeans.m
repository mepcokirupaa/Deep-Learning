function p = MKmeans(f);
 
 g = f(:, :, 2); %green channel extraction
 
 [m,n]=size(g);  % returns rows and cols of the image. m=1368, n=1712
 g=reshape(g,m*n,1); % converts 
 
 
 [cluster_idx, cluster_center] = kmeans(double(g),3,'start','uniform','EmptyAction','drop');
 % second argument indicates number of clusters to be formed. 
 %cluster_idx- index of the cluster starts from index 1st
 %cluster_center- centroid of each cluster
 % disp(cluster_idx displays the cluster id to which each pixel belongs to. rows=m*n, cols=1
 %disp(cluster_center) - displays the centroid values
 p = reshape(cluster_idx,m,n); %now m*n rows and 1 col converted to m rows and n cols
 
 x =(p==1);  % pixels with cluster_id 1 are stored in x with value set to 1. other pixels set to 0. so size will be m rows and n cols
 y =(p==2); % pixels with cluster_id 2 are stored in y with value set to 1. other pixels set to 0. so size will be m rows and n cols 
 z =(p==3);% pixels with cluster_id 3 are stored in z with value set to 1. other pixels set to 0. so size will be m rows and n cols
 figure,imshow([x,y,z]), title('After 1st Clustering'); %displays the 3 clusters
 
 R=double(f); %img copied to R
 G=double(f);
 B=double(f);
 
 x=double(x);% x , y and z are binary images with values 1 and 0
 y=double(y);
 z=double(z);
 
 
 
 %converting binary image to colored image
 R(:,:,1)=R(:,:,1).*x;  
 R(:,:,2)=R(:,:,2).*x;
 R(:,:,3)=R(:,:,3).*x;
 R=uint8(R);
 
 G(:,:,1)=G(:,:,1).*y;
 G(:,:,2)=G(:,:,2).*y;
 G(:,:,3)=G(:,:,3).*y;
 G=uint8(G);
 
 B(:,:,1)=B(:,:,1).*z;
 B(:,:,2)=B(:,:,2).*z;
 B(:,:,3)=B(:,:,3).*z;
 B=uint8(B);
 
 
 
  
 [idx]=sort(cluster_center); %sorts in ascending order the centroid values and store in idx
 if (idx(1)==cluster_center(1))  %cluster with min centroid value is given as input for second clustering
 {
  figure, imshow(R);
  MKmeanscl(x,R); %calls second kmeans clustering
 }
 elseif (idx(1)==cluster_center(2))
 {
  figure, imshow(G);
  MKmeanscl(y,G);
 }
 else
 {
  figure, imshow(B);
  MKmeanscl(z,B);
 }
end
