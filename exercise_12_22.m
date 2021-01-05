%%
%% Generate *View Consistent Light Field Superpixels* for the 
%% input light field  using the parameter settings in param
%%

colors = rand( 2000, 3);
addpath('./prop', './lines', './seg', './util','./cotton2');%검색폴더 경로 추가
D = '/home/jo/lightfieldsuperpixels/data/buddha.h5'
%D='resize.h5'
LF = HCIloadLF( D, 'X');
a=load('Disp_u.mat');
b=load('Disp_v.mat');
c=load('Seg_u.mat');
d=load('Seg_v.mat');
e=load('cl_U.mat');
f=load('cl_V.mat');
zz=load('20.mat');
gg=load('Disagree.mat');
dis=gg.X;
X=zz.X;

ss=size(D)
Du=a.Du;
Dv=b.Dv;
Epu=c.Seg_u;
Epv=d.Seg_v;
cu=e.U;
cv=f.V;
sD=size(Du);
sE=size(Epu);
sC= size(cv);
s=Dv(:,:,5);
disparityRange= [-2 2];
k=	Du(:,:,5) .* (Du(:,:,5) == Dv(:,:,5));

colors = rand( 2000, 3);
addpath('./prop', './lines', './seg', './util','./cotton2');%검색폴더 경로 추가
%D = '/home/jo/lightfieldsuperpixels/data/buddha.h5'
D='resize.h5'
LF = HCIloadLF( D, 'X');
LFu=squeeze(LF(:,:,:,:,5));
for i = 1:size(LFu, 4)% 9
  LFu(:, :, :, i) = imgaussfilt( squeeze(LFu(:, :, :, i)), 0.85);    
end
LFu = permute(LFu, [4 2 3 1]);


imshow(LF(:,:,:,5,5));

 waitforbuttonpress;
imshow(label2rgb(cu(:, :,5,5 ), colors));
 title('C U Map');
 waitforbuttonpress;
imshow(label2rgb(cv(:, :,5,5 ), colors));
 title('C V Map');
  waitforbuttonpress;
imshow(label2rgb(dis(:, :,5,5 ), colors));
 title('disagree Map');
  waitforbuttonpress;

  imshow(label2rgb(X(:, :,5,5 ), colors));
 title('superpixel Map');
  waitforbuttonpress;
for i=1:9
    for j =1:9
          imshow(label2rgb(X(:, :,i,j ), colors));
        title('VCLFS Map');
           waitforbuttonpress;

    end
end

%   
[lx,ly] = getpts
for i=1:9
    for j =1:9
        imshow(label2rgb(sD(:, :,5,5 ), colors));
        title('VCLFS Map');
           waitforbuttonpress;

    end
end

imshow(label2rgb(X(:, :,5,5 ), colors));
 title('VCLFS Map');
[lx,ly] = getpts

 
        imshow(Dv(:,:,i),disparityRange);
        title('Disparity Map');
        colormap(gca,jet) 
        colorbar
        [vx,vy] = getpts

     
        imshow(Du(:,:,i),disparityRange);
        title('Disparity Map');
        colormap(gca,jet) 
        colorbar
        [ux,uy] = getpts

     
    
     for i = 1:9
        for j=1:9
        imshow(label2rgb(cu(:, :,i,j ), colors));
        title('disagree Map');
          waitforbuttonpress;
        end
     end
    
%%%%%%% look at epi u%%%%%%%%%%%%%%%
% imshow(Du(:,:,1),disparityRange); 
%         
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
%         %[uux,uuy] = getpts
% uux
% pts=size(uuy,1);
% epi_gen=zeros(9,768);
% epc=zeros(9,768,pts);
%     for j=1:pts
%         for i=1:9
%             epi_gen(i,:)=Epu(round(uuy(j)),:,i);
%             epc(i,:,j)=Epu(round(uuy(j)),:,i);
%         end
%         epi= label2rgb(epi_gen, colors);
%         rsz=imresize(epi,[90 768]);
%         imshow(rsz)
%         impoint(gca,uux(j),9);
%         waitforbuttonpress;
%         
%         rsz=imresize(LFu(:,:,:,round(uuy(j))),[90 768]);
%         imshow(rsz)
%         impoint(gca,uux(j),9);
%      waitforbuttonpress;
%     end

% imshow(label2rgb(Epu(:,:,1), colors));

    
% epi_gen=zeros(9,768);
%     for j=1:768
%         for i=1:9
%             epi_gen(i,:)=Epu(j,:,i);
%         end
%         epi= label2rgb(epi_gen, colors)
%         rsz=imresize(epi,[90 768])
%         imshow(rsz)
%         waitforbuttonpress;
%     end


% epi_gen=zeros(9,768);
%     for j=1:768
%         for i=1:9
%             epi_gen(i,:)=Epu(j,:,i);
%         end
%         epi= label2rgb(epi_gen, colors);
%         rsz=imresize(epi,[90 768]);
%         imshow(rsz)
%         j
%         waitforbuttonpress;
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imshow(k,disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
%          waitforbuttonpress;
% imshow(label2rgb(Epu(:, :,5 ), colors));
%  title('C U Map');
%  waitforbuttonpress;
% imshow(label2rgb(Epv(:, :,5 ), colors));
%  title('C V Map');
%  waitforbuttonpress;
% imshow(LF(:,:,:,5,5));
%  waitforbuttonpress;
%  imshow(label2rgb(Epu(:, :,5 ), colors));
%  title('C U Map');
%  waitforbuttonpress;
% imshow(label2rgb(Epv(:, :,5 ), colors));
%  title('C V Map');
%  waitforbuttonpress;
% imshow(LF(:,:,:,5,5));
%  waitforbuttonpress;
% imshow(Du(:,:,5),disparityRange); 
%         
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
%          %[uux,uuy] = getpts
%         
% imshow(Dv(:,:,5),disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
%         %[vvx,vvy] = getpts
% 
% uu=size(ux,1);
% vv=size(vx,1);
% 
% 
%   imshow(LF(:,:,:,5,5));
%   for i=1:vv
%   impoint(gca,vx(i),vy(i));
%   end
%   plt(vy,vx);  
%   waitforbuttonpress;
%   imshow(Dv(:,:,5),disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
%         [vx,vy] = getpts
        
        
        
        
function plt(py,px)
ps=size(py,1);
  for i=1:ps
  impoint(gca,px(i),py(i));
  end
end
function epiview(LF)
%%%%%%%%%%%%%%%%%%original EPI view%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LFu=squeeze(LF(:,:,:,:,5));
for i = 1:size(LFu, 4)% 9
  LFu(:, :, :, i) = imgaussfilt( squeeze(LFu(:, :, :, i)), 0.85);    
end
LFu = permute(LFu, [4 2 3 1]);

LFv=squeeze(LF(:,:,:,5,:));
LFv = permute(LFv, [2 1 3 4]);
 for i = 1:size(LFv, 4) 
    LFv(:, :, :, i) = imgaussfilt( squeeze(LFv(:, :, :, i)), 0.85);
  end
 LFv = permute(LFv, [2 4 3 1]);
 size (LFv)
for i=1:768
    
     rsz=imresize(LFu(:,:,:,i),[90 768]);
     imshow(rsz)
     i
     waitforbuttonpress;
end
 for i=1:768
    
     rsz=imresize(LFv(:,:,:,i),[768 90]);
     imshow(rsz)
     i
     waitforbuttonpress;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


%D='papillon.h5'

%      for i = 1:9
%         for j=1:9
%         imshow(label2rgb(X(:, :,i,j ), colors));
%         title('disagree Map');
%           waitforbuttonpress;
%         end
%     end

