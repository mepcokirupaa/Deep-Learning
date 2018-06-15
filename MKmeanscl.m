function BW2 = MKmeanscl(p,f)
 
 [r,c]=size(p);
 p=reshape(p,r*c,1);
 
 [c1,v1] = kmeans(double(p),2,'start','uniform','EmptyAction','drop');
 R=reshape(c1,r,c);
 
 x1=(R==1);
 y1=(R==2);
 
 figure,imshow([x1,y1]), title('After 2nd Clustering');
 
 [idx]=sort(v1);
 if (v1(1)==idx(2))
 {
  figure, imshow(x1), title('Nucleus');
  DoWatershed(x1);
 }
 elseif (v1(2)==idx(2))
 {
  figure, imshow(y1), title('Nucleus');
  DoWatershed(y1);
 }
end