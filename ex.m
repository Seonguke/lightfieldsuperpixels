% filename_pfm='/home/jo/바탕화면/new_HCI_DATA/full_data/training/dino/gt_disp_highres.pfm';
% 
% fid = fopen(filename_pfm);
% 
% fscanf(fid,'%c',[1,3]);
% cols = fscanf(fid,'%f',1);
% rows = fscanf(fid,'%f',1);
% fscanf(fid,'%f',1);
% fscanf(fid,'%c',1);
% D = fread(fid,[cols,rows],'single');
% D(D == Inf) = 0;
% D = rot90(D);
%     disparityRange = [-2 2];
%         imshow(D,disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet) 
%         colorbar
% 
% fclose(fid);
LightField_RefName = 'antinous';
Z=zeros(512,512,9,9);
for i = 1:81
    View_RefName{1,i} = fullfile(LightField_RefName,['gt_disp_lowres_Cam' num2str(i-1,'%03d') '.pfm']);
    fid = fopen(View_RefName{1,i});

    fscanf(fid,'%c',[1,3]);
    cols = fscanf(fid,'%f',1);
    rows = fscanf(fid,'%f',1);
    fscanf(fid,'%f',1);
    fscanf(fid,'%c',1);
    D = fread(fid,[cols,rows],'single');
    D(D == Inf) = 0;
    D = rot90(D);
   
    Q = fix((i-1)/9)+1;
    R=rem((i-1),9)+1;
    Z(:,:,Q,R)=D;
end
   disparityRange = [-2 2];
% for i=1:9
%     for j=1:9
% 
%    
%         imshow(X(:,:,i,j),disparityRange);
%         title('Disparity Map');
%         colormap(gca,jet); 
%         colorbar;
%      waitforbuttonpress;
%     end
% end
addpath('./prop', './lines', './seg', './util');%검색폴더 경로 추가


X=zeros(size(Z));
    for i = 1:9
        for j=1:9
        asd=abs(10-j);
        X(:,:,i,j)=Z(:,:,i,asd);
        end
    end
D = '/home/jo/PycharmProjects/pythonProject2/additional/antinous/antinous.h5'

LF = HCIloadLF( D, 'x');
size(LF)
save(['/home/jo/lightfieldsuperpixels/custom/Dgt.mat'], 'X');