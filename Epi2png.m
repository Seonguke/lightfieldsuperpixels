% epi_mat=load('epi.mat');
% G=(gpuArray(epi_mat.s))%mat형식 sturct니깐
% 
% cnt=size(G(:,1))
% szRow=cnt(1);
% epi_idx=G(end,5)
%%%%%%%%%%%%% function으로 만들어야댈듯~%%%%%%%%%%%%%%%%%%%%%%

%parfor i=1:szRow
    
%end
% 
% suu= load('Su.mat');
% svu= load('Sv.mat');
% a=cell2mat(suu.Su(1))
% seg_idx=size(cell2mat(suu.Su(1)),1)
% C = ones(512,512,9).*0.5; 
%C(:,:,1) = [.1 .2 .3; .4 .5 .6; .7 .8 .9]
mask = load('epi_label.mat')
label=mask.labels
ep_mask=zeros(512,512,9)
for i =1:9
    ep_mask(1,:,i)=label(i,:)
end


segmid=ep_mask(1,:,5)
% image(C(:,:,5))
% parfor i=1 : 512;%병렬 연산
%    
%     if isempty(cell2mat(suu.Su(i)))
%         %disp('empty')
%         continue
%     else
%         seg_idx=size(cell2mat(suu.Su(i)),1)
%         disp(i)
%     end
% end
% su=suu.Su
% su
% sv
% n = zeros(2, 1);
% n(1) = sum(cellfun(@(s) size(s, 1), Su));
% n(2) = sum(cellfun(@(s) size(s, 1), Su));