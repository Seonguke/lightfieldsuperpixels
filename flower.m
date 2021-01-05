
zz=load('/home/jo/lightfieldsuperpixels/custom/Dgt.mat');
lfr=zz.X;

    disparityRange = [-2 2];
for i =1:9
    
    for j=1:9
        waitforbuttonpress;
    title('Disparity Map');
    colormap(gca,jet) 
    colorbar
        imshow(lfr(:,:,i,j),disparityRange);
  

    end
end

 lfr=squeeze(LF(5,5,:,:,4));

% for i=1:14
%     for j=1:14
%         lfrr=squeeze(LF(i,j,:,:,1:3));
%        
%         c=strcat('/home/jo/flower/flow/img_',num2str(i),'_',num2str(j),'.png')
%         
%         imshow(squeeze(LF(i,j,:,:,1:3)))
%          imwrite(lfrr,c);
%     end
% end
